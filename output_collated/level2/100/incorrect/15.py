# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_120759/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'min_value', 'divisor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'min_value', 'divisor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

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
    # State for conv_transpose (nn.ConvTranspose3d)
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
    if 'min_value' in flat_state:
        state_kwargs['min_value'] = flat_state['min_value']
    else:
        state_kwargs['min_value'] = getattr(model, 'min_value')
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
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

# CUDA source – contains the fused 3D transposed-conv + clamp + scale kernel
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int K,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const float min_val,
    const float div_factor,
    float* __restrict__ output)
{
    const long long total_elements = (long long)N * C_out * D_out * H_out * W_out;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    // Decode linear index
    long long tmp = idx;
    int w_out = tmp % W_out; tmp /= W_out;
    int h_out = tmp % H_out; tmp /= H_out;
    int d_out = tmp % D_out; tmp /= D_out;
    int c_out = tmp % C_out; tmp /= C_out;
    int n = tmp;

    float acc = bias[c_out];

    // Transposed Conv Logic
    // Output element (n, c_out, d_out, h_out, w_out) receives contributions 
    // from input (n, c_in, d_in, h_in, w_in) if d_out = d_in * stride + kd - pad
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kd = 0; kd < K; ++kd) {
            int d_in = d_out + pad_d - kd;
            if (d_in < 0 || d_in % stride_d != 0) continue;
            d_in /= stride_d;
            if (d_in >= D_in) continue;

            for (int kh = 0; kh < K; ++kh) {
                int h_in = h_out + pad_h - kh;
                if (h_in < 0 || h_in % stride_h != 0) continue;
                h_in /= stride_h;
                if (h_in >= H_in) continue;

                for (int kw = 0; kw < K; ++kw) {
                    int w_in = w_out + pad_w - kw;
                    if (w_in < 0 || w_in % stride_w != 0) continue;
                    w_in /= stride_w;
                    if (w_in >= W_in) continue;

                    // weight index: [C_in, C_out, K, K, K] -> flattened
                    // PyTorch conv_transpose weight order: [C_in, C_out, K, K, K]
                    int weight_idx = c_in * (C_out * K * K * K) + c_out * (K * K * K) + (kd * K * K + kh * K + kw);
                    int in_idx = n * (C_in * D_in * H_in * W_in) + c_in * (D_in * H_in * W_in) + d_in * (H_in * W_in) + h_in * W_in + w_in;
                    
                    acc += input[in_idx] * weight[weight_idx];
                }
            }
        }
    }

    acc = (acc < min_val) ? min_val : acc;
    output[idx] = acc / div_factor;
}

void launch_fused_conv(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
    torch::Tensor& output,
    int stride, int pad, float min_val, float div_factor)
{
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);
    const int C_out = weight.size(1);
    const int K = weight.size(2);
    const int D_out = output.size(2);
    const int H_out = output.size(3);
    const int W_out = output.size(4);

    const long long total_elements = (long long)N * C_out * D_out * H_out * W_out;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    fused_conv_transpose_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out, K,
        stride, stride, stride, pad, pad, pad, min_val, div_factor, output.data_ptr<float>()
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_fused_conv(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, 
                       torch::Tensor& output, int stride, int pad, float min_val, float div_factor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused_conv, "Fused Transpose Conv");
}
"""

module = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_transpose_weight,
    conv_transpose_bias=None,
    conv_transpose_stride=2,
    conv_transpose_padding=1,
    conv_transpose_output_padding=0,
    conv_transpose_groups=1,
    conv_transpose_dilation=1,
    min_value=-1.0,
    divisor=2.0,
):
    # Note: Assumes kernel is cubic and stride/padding are scalars as in the provided snippet example context
    N, C_in, D_in, H_in, W_in = x.shape
    C_in_w, C_out, K, _, _ = conv_transpose_weight.shape
    
    stride = conv_transpose_stride if isinstance(conv_transpose_stride, int) else conv_transpose_stride[0]
    pad = conv_transpose_padding if isinstance(conv_transpose_padding, int) else conv_transpose_padding[0]
    
    D_out = (D_in - 1) * stride - 2 * pad + K
    H_out = (H_in - 1) * stride - 2 * pad + K
    W_out = (W_in - 1) * stride - 2 * pad + K
    
    output = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    bias = conv_transpose_bias if conv_transpose_bias is not None else torch.zeros(C_out, device=x.device, dtype=x.dtype)
    
    module.fused_op(x.contiguous(), conv_transpose_weight.contiguous(), bias.contiguous(), output, stride, pad, min_value, divisor)
    return output
