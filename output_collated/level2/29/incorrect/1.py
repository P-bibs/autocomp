# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_103731/code_5.py
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

# The naive implementation of matrix multiplication in the previous step
# is suboptimal for large matrices. However, to meet the requirement of
# replacing built-in matmul kernels with custom CUDA, we utilize a 
# specialized kernel structure.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float mish(float x) {
    // Mish = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    return x * tanhf(logf(1.0f + expf(expf(x) > 20.0f ? x : logf(1.0f + expf(x)))));
}

// Tiled matrix multiplication kernel with fused double-mish activation
template <int BLOCK_SIZE>
__global__ void fused_linear_mish_kernel(const float* __restrict__ x, 
                                        const float* __restrict__ weight, 
                                        const float* __restrict__ bias, 
                                        float* __restrict__ out, 
                                        int M, int N, int K) {
    
    __shared__ float s_x[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_w[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        if (row < M && (t * BLOCK_SIZE + threadIdx.x) < N)
            s_x[threadIdx.y][threadIdx.x] = x[row * N + t * BLOCK_SIZE + threadIdx.x];
        else
            s_x[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < K && (t * BLOCK_SIZE + threadIdx.y) < N)
            s_w[threadIdx.y][threadIdx.x] = weight[col * N + t * BLOCK_SIZE + threadIdx.y];
        else
            s_w[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += s_x[threadIdx.y][k] * s_w[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        sum += bias[col];
        float m1 = mish(sum);
        out[row * K + col] = mish(m1);
    }
}

void fused_op(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out) {
    const int M = x.size(0);
    const int N = x.size(1);
    const int K = weight.size(0);
    const int BLOCK_SIZE = 32;

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((K + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    fused_linear_mish_kernel<BLOCK_SIZE><<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), M, N, K
    );
}
"""

cpp_source = r"""
void fused_op(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused Linear + Double Mish");
}
"""

fused_lib = load_inline(
    name='fused_lib',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias):
    # linear_weight is (out_features, in_features)
    out = torch.empty(x.size(0), linear_weight.size(0), device=x.device, dtype=x.dtype)
    fused_lib.fused_op(x, linear_weight, linear_bias, out)
    return out

# Global parameter setup for the evaluator
batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda')]
