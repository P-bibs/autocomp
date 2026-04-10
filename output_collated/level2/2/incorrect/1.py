# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_161645/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape', 'scaling_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias', 'scaling_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, *args) -> torch.Tensor:
        return functional_model(*args, **extract_state_kwargs(self))


def build_reference_model():
    init_inputs = list(get_init_inputs())
    model = ModelNew(*init_inputs)
    model.eval()
    return model


def extract_state_kwargs(model):
    flat_state = {}
    for name, value in model.named_parameters():
        flat_state[name.replace('.', '_')] = value
    for name, value in model.named_buffers():
        flat_state[name.replace('.', '_')] = value
    state_kwargs = {}
    init_inputs = list(get_init_inputs())
    init_arg_map = {name: value for name, value in zip(INIT_PARAM_NAMES, init_inputs)}
    # State for conv_transpose (nn.ConvTranspose2d)
    if 'conv_transpose_weight' in flat_state:
        state_kwargs['conv_transpose_weight'] = flat_state['conv_transpose_weight']
    else:
        state_kwargs['conv_transpose_weight'] = getattr(model.conv_transpose, 'weight', None)
    if 'conv_transpose_bias' in flat_state:
        state_kwargs['conv_transpose_bias'] = flat_state['conv_transpose_bias']
    else:
        state_kwargs['conv_transpose_bias'] = getattr(model.conv_transpose, 'bias', None)
    state_kwargs['conv_transpose_stride'] = model.conv_transpose.stride
    state_kwargs['conv_transpose_padding'] = model.conv_transpose.padding
    state_kwargs['conv_transpose_output_padding'] = model.conv_transpose.output_padding
    state_kwargs['conv_transpose_groups'] = model.conv_transpose.groups
    state_kwargs['conv_transpose_dilation'] = model.conv_transpose.dilation
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    if 'scaling_factor' in flat_state:
        state_kwargs['scaling_factor'] = flat_state['scaling_factor']
    else:
        state_kwargs['scaling_factor'] = getattr(model, 'scaling_factor')
    missing = [name for name in REQUIRED_STATE_NAMES if name not in state_kwargs]
    if missing:
        raise RuntimeError(f'Missing required state entries: {missing}')
    return state_kwargs


def get_functional_inputs():
    model = build_reference_model()
    forward_args = tuple(get_inputs())
    state_kwargs = extract_state_kwargs(model)
    return forward_args, state_kwargs




import torch
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# Optimized CUDA Kernel: Fused Transposed Conv + Bias + Clamp
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_clamp_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float inv_scaling_factor,
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in,
    const int H_out, const int W_out,
    const int stride, const int padding, const int K,
    float* __restrict__ out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * H_out * W_out) return;

    // Decode linear index to (n, oc, oh, ow)
    int n   = idx / (C_out * H_out * W_out);
    int rem = idx % (C_out * H_out * W_out);
    int oc  = rem / (H_out * W_out);
    int oh  = (rem / W_out) % H_out;
    int ow  = rem % W_out;

    float sum = 0.0f;
    const int weight_oc_offset = oc * C_in * K * K;

    // Naive Transposed Conv loops (Direct summation approach)
    for (int ic = 0; ic < C_in; ++ic) {
        const int weight_ic_offset = weight_oc_offset + (ic * K * K);
        const int x_base = ((n * C_in + ic) * H_in);

        for (int kh = 0; kh < K; ++kh) {
            int ih = (oh + padding - kh);
            if (ih < 0 || ih % stride != 0) continue;
            ih /= stride;
            if (ih >= H_in) continue;

            for (int kw = 0; kw < K; ++kw) {
                int iw = (ow + padding - kw);
                if (iw < 0 || iw % stride != 0) continue;
                iw /= stride;
                if (iw >= W_in) continue;

                sum += x[x_base * W_in + (ih * W_in + iw)] * weight[weight_ic_offset + (kh * K + kw)];
            }
        }
    }

    sum += bias[oc];

    // Fused Scaling Logic: x -> clamp(x, 0, 1) -> * scale -> clamp(x, 0, 1) -> / scale
    // This is equivalent to clamping the original value to [0, 1/scaling_factor]
    if (sum < 0.0f) sum = 0.0f;
    else if (sum > inv_scaling_factor) sum = inv_scaling_factor;

    out[idx] = sum;
}

void fused_op_forward(
    const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& bias,
    float inv_scaling, int stride, int padding, int K, torch::Tensor& out)
{
    const int N = x.size(0);
    const int C_in = x.size(1);
    const int C_out = weight.size(1);
    const int H_in = x.size(2);
    const int W_in = x.size(3);
    const int H_out = out.size(2);
    const int W_out = out.size(3);

    const int total_threads = N * C_out * H_out * W_out;
    const int threads = 256;
    const int blocks = (total_threads + threads - 1) / threads;

    fused_conv_transpose_clamp_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        inv_scaling, N, C_in, C_out, H_in, W_in, H_out, W_out,
        stride, padding, K, out.data_ptr<float>());
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& bias, float inv_scaling, int stride, int padding, int K, torch::Tensor& out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Transposed Conv kernel");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(
    x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride,
    conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups,
    conv_transpose_dilation, bias, scaling_factor,
):
    H_in, W_in = x.shape[2], x.shape[3]
    K = conv_transpose_weight.shape[2]
    H_out = (H_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + (K - 1) + conv_transpose_output_padding + 1
    W_out = (W_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + (K - 1) + conv_transpose_output_padding + 1
    
    combined_bias = (conv_transpose_bias.flatten() + bias.flatten()).contiguous()
    out = torch.empty((x.size(0), conv_transpose_weight.size(1), H_out, W_out), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(
        x.contiguous(), conv_transpose_weight.permute(0, 1, 2, 3).contiguous(), 
        combined_bias, 1.0 / scaling_factor, conv_transpose_stride, 
        conv_transpose_padding, K, out
    )
    return out
