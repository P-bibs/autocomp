# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_154319/code_15.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'add_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'add_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

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
    if 'add_value' in flat_state:
        state_kwargs['add_value'] = flat_state['add_value']
    else:
        state_kwargs['add_value'] = getattr(model, 'add_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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
# Inline CUDA source – fused transposed‑convolution + point‑wise ops
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fast gelu approximation: 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
__device__ __forceinline__ float fast_gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    float t = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(t));
}

__global__ void deconv_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in,
    const int H_out, const int W_out,
    const int K, const int stride,
    const int padding, const int dilation,
    const int groups,
    const float add_value,
    const float multiply_value)
{
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_out = N * C_out * H_out * W_out;
    if (out_idx >= total_out) return;

    // Decode output layout NCHW
    int idx = out_idx;
    const int ow = idx % W_out; idx /= W_out;
    const int oh = idx % H_out; idx /= H_out;
    const int oc = idx % C_out; idx /= C_out;
    const int n  = idx;

    const int out_ch_per_group = C_out / groups;
    const int in_ch_per_group  = C_in  / groups;
    const int g = oc / out_ch_per_group;

    float sum = (bias != nullptr) ? bias[oc] : 0.0f;

    // Transposed convolution implementation: 
    // Accumulate contributions from overlapping sliding windows
    for (int ic = 0; ic < in_ch_per_group; ++ic) {
        const int ic_global = g * in_ch_per_group + ic;
        const int w_base = (ic_global * out_ch_per_group + (oc % out_ch_per_group)) * K * K;

        #pragma unroll
        for (int kh = 0; kh < K; ++kh) {
            const int ihi = oh + padding - kh * dilation;
            if (ihi % stride != 0 || ihi < 0 || ihi >= H_in * stride) continue;
            const int ih = ihi / stride;

            #pragma unroll
            for (int kw = 0; kw < K; ++kw) {
                const int iwi = ow + padding - kw * dilation;
                if (iwi % stride != 0 || iwi < 0 || iwi >= W_in * stride) continue;
                const int iw = iwi / stride;

                const float w = weight[w_base + kh * K + kw];
                const float val = input[((n * C_in + ic_global) * H_in + ih) * W_in + iw];
                sum += val * w;
            }
        }
    }

    // Fused point-wise operations
    sum += add_value;
    sum = fminf(sum, 0.0f);
    sum = fast_gelu(sum);
    output[out_idx] = sum * multiply_value;
}

void deconv_fused(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in,
    const int H_out, const int W_out,
    const int K, const int stride,
    const int padding, const int dilation,
    const int groups,
    const float add_value,
    const float multiply_value)
{
    const int threads = 256;
    const int total_out = N * C_out * H_out * W_out;
    const int blocks = (total_out + threads - 1) / threads;

    const float* bias_ptr = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;

    deconv_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
        N, C_in, C_out, H_in, W_in, H_out, W_out,
        K, stride, padding, dilation, groups,
        add_value, multiply_value);
}
"""

cpp_source = r"""
void deconv_fused(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& output,
                  const int N, const int C_in, const int C_out, const int H_in, const int W_in, 
                  const int H_out, const int W_out, const int K, const int stride, const int padding, 
                  const int dilation, const int groups, const float add_value, const float multiply_value);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("deconv_fused", &deconv_fused, "Fused deconv kernel");
}
"""

_module = None
def _get_module():
    global _module
    if _module is None:
        _module = load_inline(name="fused_deconv", cpp_sources=cpp_source, cuda_sources=cuda_source, 
                              extra_cuda_cflags=["-O3", "--use_fast_math"], with_cuda=True)
    return _module

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, add_value, multiply_value):
    N, C_in, H_in, W_in = x.shape
    C_out = conv_transpose_weight.shape[1] * conv_transpose_groups
    K = conv_transpose_weight.shape[2]
    
    H_out = (H_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    W_out = (W_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    
    out = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    _get_module().deconv_fused(
        x.contiguous(), conv_transpose_weight.contiguous(), 
        conv_transpose_bias.contiguous() if conv_transpose_bias is not None else torch.tensor([], device=x.device),
        out, N, C_in, C_out, H_in, W_in, H_out, W_out, K, conv_transpose_stride, 
        conv_transpose_padding, conv_transpose_dilation, conv_transpose_groups, add_value, multiply_value
    )
    return out
