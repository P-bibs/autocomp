# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_052603/code_4.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies HardSwish, and then ReLU.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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
    # State for conv (nn.Conv2d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
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

# CUDA kernel implementation
# Note: This is an un-tiled implementation. For extreme performance in convolutions, 
# shared memory tiling (storing input patches) is typically used.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float hardswish_relu_fused(float x) {
    float hswish = x * fmaxf(0.0f, fminf(x + 3.0f, 6.0f)) / 6.0f;
    return fmaxf(0.0f, hswish);
}

__global__ void fused_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int C_out,
    int H, int W, int K,
    int S, int P, int D,
    int H_out, int W_out) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C_out * H_out * W_out;
    if (tid >= total_elements) return;

    int n = tid / (C_out * H_out * W_out);
    int c_out = (tid / (H_out * W_out)) % C_out;
    int h_out = (tid / W_out) % H_out;
    int w_out = tid % W_out;

    float val = bias[c_out];
    
    // Manual convolution loop
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int h_in = h_out * S - P + kh * D;
                int w_in = w_out * S - P + kw * D;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    val += input[((n * C_in + c_in) * H + h_in) * W + w_in] * 
                           weight[(((c_out * C_in + c_in) * K + kh) * K + kw)];
                }
            }
        }
    }
    
    output[tid] = hardswish_relu_fused(val);
}

void launch_fused_conv(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
    torch::Tensor& output, int S, int P, int D) {
    
    int B = input.size(0); int C_in = input.size(1);
    int H = input.size(2); int W = input.size(3);
    int C_out = weight.size(0); int K = weight.size(2);
    int H_out = output.size(2); int W_out = output.size(3);

    int total = B * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_conv_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), B, C_in, C_out, H, W, K, S, P, D, H_out, W_out
    );
}
"""

cpp_source = r"""
void launch_fused_conv(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& output, int S, int P, int D);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &launch_fused_conv, "Fused Conv+Hardswish+Relu");
}
"""

fused_ext = load_inline(
    name='fused_conv', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    # This implementation assumes groups=1 as per requirement.
    B, C_in, H, W = x.shape
    C_out, _, K, _ = conv_weight.shape
    H_out = (H + 2 * conv_padding - conv_dilation * (K - 1) - 1) // conv_stride + 1
    W_out = (W + 2 * conv_padding - conv_dilation * (K - 1) - 1) // conv_stride + 1
    
    output = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    fused_ext.fused_conv(x, conv_weight, conv_bias, output, conv_stride, conv_padding, conv_dilation)
    return output
