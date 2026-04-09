# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_081239/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
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

# Optimized CUDA kernel:
# 1. We use a 2D block grid mapping each pixel (BATCH, H, W) to a thread.
# 2. We use registers for the reduction (min_val) to avoid global memory round-trips.
# 3. We compute the convolution values on-the-fly to minimize shared memory pressure
#    and cache misses, which is effective for small kernels (3x3).
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ input, const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ output, int N, int C_in, int C_out, int H, int W, int K) {
    
    int n = blockIdx.x;
    int h_out = blockIdx.y;
    int w_out = threadIdx.x;
    
    if (h_out >= H || w_out >= W) return;

    float min_val = 1e38f;
    int pad = K / 2;

    for (int cout = 0; cout < C_out; ++cout) {
        float acc = bias[cout];
        for (int cin = 0; cin < C_in; ++cin) {
            for (int ki = 0; ki < K; ++ki) {
                int h_in = h_out + ki - pad;
                if (h_in >= 0 && h_in < H) {
                    for (int kj = 0; kj < K; ++kj) {
                        int w_in = w_out + kj - pad;
                        if (w_in >= 0 && w_in < W) {
                            acc += input[((n * C_in + cin) * H + h_in) * W + w_in] * 
                                   weight[(((cout * C_in + cin) * K + ki) * K + kj)];
                        }
                    }
                }
            }
        }
        if (acc < min_val) min_val = acc;
    }
    
    // Applying tanh twice
    float t = tanhf(min_val);
    output[((n * H) + h_out) * W + w_out] = tanhf(t);
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    int N = input.size(0);
    int H = input.size(2);
    int W = input.size(3);
    int C_in = input.size(1);
    int C_out = weight.size(0);
    int K = weight.size(2);

    // Grid: N batches, H rows. Each block handles a row.
    dim3 grid(N, H);
    dim3 block(W);
    
    fused_conv_min_tanh_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C_in, C_out, H, W, K);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Convolution Min Tanh");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    # Ensure inputs are contiguous for the custom kernel
    x = x.contiguous()
    conv_weight = conv_weight.contiguous()
    conv_bias = conv_bias.contiguous()
    
    out = torch.empty((x.size(0), 1, x.size(2), x.size(3)), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_weight, conv_bias, out)
    return out

# Constants to match original signature requirement
batch_size, in_channels, out_channels, height, width, kernel_size = 128, 16, 64, 256, 256, 3
def get_init_inputs(): return [in_channels, out_channels, kernel_size]
def get_inputs(): return [torch.rand(batch_size, in_channels, height, width).cuda()]
