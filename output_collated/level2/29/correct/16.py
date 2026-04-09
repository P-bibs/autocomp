# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_104842/code_7.py
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

# -------------------------------------------------------------------------
# CUDA source – Tiled GEMM with fused double-mish and bias addition
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float mish(float x) {
    // Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    // __expf and __logf are fast math intrinsics
    return x * tanhf(__logf(1.0f + __expf(x)));
}

__global__ void gemm_mish_fused_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    const float* __restrict__ bias, 
    float* __restrict__ C, 
    int M, int N, int K) 
{
    // Block dimensions for tiles
    const int TILE_SIZE = 16;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over input tiles
    for (int k = 0; k < (K + TILE_SIZE - 1) / TILE_SIZE; ++k) {
        // Load Input A (M x K)
        int a_col = k * TILE_SIZE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        // Load Weights B (N x K) -> accessed transposed as (K x N)
        // B is provided as (out_features, in_features), i.e., N x K
        int b_row = k * TILE_SIZE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (col < N && b_row < K) ? B[col * K + b_row] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        float val = sum + (bias ? bias[col] : 0.0f);
        // Apply Mish twice while result is in register
        float m1 = mish(val);
        float m2 = mish(m1);
        C[row * N + col] = m2;
    }
}

void launch_gemm_mish_fused(
    at::Tensor A, at::Tensor B, at::optional<at::Tensor> bias, at::Tensor C)
{
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    dim3 threads(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    gemm_mish_fused_kernel<<<grid, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), bias_ptr, C.data_ptr<float>(), M, N, K
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_gemm_mish_fused(
    at::Tensor A, at::Tensor B, at::optional<at::Tensor> bias, at::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_gemm_mish_fused, "Fused GEMM + 2x Mish kernel");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias):
    """
    Computes y = mish(mish(x @ W.T + b)) using a single fused CUDA kernel.
    """
    # Ensure contiguous memory layout for kernel access
    x = x.contiguous()
    w = linear_weight.contiguous()
    b = linear_bias.contiguous() if linear_bias is not None else None
    
    # Output tensor: (batch_size, out_features)
    out = torch.empty((x.size(0), w.size(0)), device=x.device, dtype=x.dtype)
    
    # Execute custom kernel
    fused_ext.fused_op(x, w, b, out)
    return out
