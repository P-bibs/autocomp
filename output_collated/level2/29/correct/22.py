# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110005/code_5.py
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

# The naive implementation in the prompt is replaced with a tiled CUDA kernel.
# A simple row-dot-product loop is inefficient for global memory access.
# Using shared memory tiling significantly improves memory bandwidth utilization for GEMM.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 32

__device__ __forceinline__ float mish_fast(float x) {
    // f(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    // For large x, softplus(x) -> x, tanh(x) -> 1. For small, handle exp carefully.
    float softplus = logf(1.0f + expf(x));
    return x * tanhf(softplus);
}

__global__ void fused_linear_mish_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ out, 
    int B, int In, int Out) {
    
    __shared__ float s_x[TILE_SIZE][TILE_SIZE];
    __shared__ float s_w[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (In + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < B && (t * TILE_SIZE + threadIdx.x) < In)
            s_x[threadIdx.y][threadIdx.x] = x[row * In + t * TILE_SIZE + threadIdx.x];
        else
            s_x[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < Out && (t * TILE_SIZE + threadIdx.y) < In)
            s_w[threadIdx.y][threadIdx.x] = weight[col * In + t * TILE_SIZE + threadIdx.y];
        else
            s_w[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += s_x[threadIdx.y][k] * s_w[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < B && col < Out) {
        acc += bias[col];
        float m1 = mish_fast(acc);
        out[row * Out + col] = mish_fast(m1);
    }
}

void fused_linear_mish(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out) {
    int B = x.size(0);
    int In = x.size(1);
    int Out = weight.size(0);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((Out + TILE_SIZE - 1) / TILE_SIZE, (B + TILE_SIZE - 1) / TILE_SIZE);

    fused_linear_mish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        out.data_ptr<float>(), B, In, Out
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_linear_mish(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_mish", &fused_linear_mish, "Fused Linear and Mish activation");
}
"""

fused_ext = load_inline(
    name='fused_linear_mish_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias):
    # Weight provided is [out_features, in_features]
    out = torch.empty((x.size(0), linear_weight.size(0)), device=x.device, dtype=x.dtype)
    fused_ext.fused_linear_mish(x, linear_weight, linear_bias, out)
    return out

batch_size, in_features, out_features = 1024, 8192, 8192
def get_init_inputs(): return [in_features, out_features]
def get_inputs(): return [torch.rand(batch_size, in_features, device='cuda')]
