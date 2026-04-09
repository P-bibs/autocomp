# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_104842/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['linear_weight', 'linear_bias']
REQUIRED_FLAT_STATE_NAMES = ['linear_weight', 'linear_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Mish, and applies Mish again.
    """

    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

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
    # State for linear (nn.Linear)
    if 'linear_weight' in flat_state:
        state_kwargs['linear_weight'] = flat_state['linear_weight']
    else:
        state_kwargs['linear_weight'] = getattr(model.linear, 'weight', None)
    if 'linear_bias' in flat_state:
        state_kwargs['linear_bias'] = flat_state['linear_bias']
    else:
        state_kwargs['linear_bias'] = getattr(model.linear, 'bias', None)
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

# CUDA Kernel: Fused Linear + Mish (Redundant Mish collapsed)
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float mish(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

// Simple tiled kernel to replace torch.matmul + bias + activation
__global__ void fused_linear_mish_kernel(const float* __restrict__ input, 
                                        const float* __restrict__ weight, 
                                        const float* __restrict__ bias, 
                                        float* __restrict__ output,
                                        int batch_size, int in_features, int out_features) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        sum += bias[col];
        // Original code applied mish twice. Mathematically, the kernel calculates Mish(Mish(x)).
        // We fuse the computation to avoid multiple global memory roundtrips.
        output[row * out_features + col] = mish(mish(sum));
    }
}

void fused_linear_mish_forward(int blocks_x, int blocks_y, int threads_x, int threads_y,
                               torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    dim3 threads(threads_x, threads_y);
    dim3 blocks(blocks_x, blocks_y);
    fused_linear_mish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.data_ptr<float>(), output.data_ptr<float>(), 
        input.size(0), input.size(1), weight.size(0));
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_linear_mish_forward(int blocks_x, int blocks_y, int threads_x, int threads_y,
                               torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_mish_forward", &fused_linear_mish_forward, "Fused Linear + Mish forward pass");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias):
    batch_size = x.size(0)
    out_features = linear_weight.size(0)
    output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
    
    # Use the fused kernel instead of torch.nn.functional calls
    fused_ext.fused_linear_mish_forward(
        (out_features + 31) // 32,  # blocks_x
        (batch_size + 15) // 16,    # blocks_y
        32,                         # threads_x
        16,                         # threads_y
        x, linear_weight, linear_bias, output
    )
    return output

# Setup for evaluation
batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda')]
