# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_104237/code_7.py
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

# ----------------------------------------------------------------------
# CUDA Kernel: Tiled GEMM with Fused 2x Mish
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 16

__device__ __forceinline__ float mish(float x) {
    // mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    // using fast_math primitives
    return x * tanhf(log1pf(expf(x)));
}

__global__ void fused_linear_mish_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int k_step = 0; k_step < (K + TILE_SIZE - 1) / TILE_SIZE; ++k_step) {
        // Load tile A
        if (row < M && (k_step * TILE_SIZE + tx) < K)
            As[ty][tx] = A[row * K + (k_step * TILE_SIZE + tx)];
        else
            As[ty][tx] = 0.0f;

        // Load tile B (B is K x N)
        if ((k_step * TILE_SIZE + ty) < K && col < N)
            Bs[ty][tx] = B[(k_step * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        sum += bias[col];
        // Apply mish twice
        float val = mish(sum);
        val = mish(val);
        C[row * N + col] = val;
    }
}

void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor bias, torch::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    fused_linear_mish_kernel<<<grid, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), bias.data_ptr<float>(), 
        C.data_ptr<float>(), M, N, K
    );
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor bias, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Linear + 2x Mish");
}
"""

# Compile the inline extension
fused_module = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias):
    # Prepare inputs: Ensure contiguity for kernel indexing
    # Original linear is Y = x @ W.t() + b
    # Kernel expects A(M, K) @ B(K, N) + b
    # So B must be W.t()
    B = linear_weight.t().contiguous()
    bias = linear_bias if linear_bias is not None else torch.zeros(linear_weight.size(0), device=x.device)
    
    M, K = x.shape
    N = B.size(1)
    
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    fused_module.fused_op(x, B, bias, output)
    return output

def get_init_inputs():
    return [8192, 8192]

def get_inputs():
    return [torch.rand(1024, 8192, device='cuda')]
