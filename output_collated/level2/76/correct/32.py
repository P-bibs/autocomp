# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_024220/code_15.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'bias_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['gemm_weight', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['gemm_weight', 'bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a bias term, and applies ReLU.
    """

    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

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
    # State for gemm (nn.Linear)
    if 'gemm_weight' in flat_state:
        state_kwargs['gemm_weight'] = flat_state['gemm_weight']
    else:
        state_kwargs['gemm_weight'] = getattr(model.gemm, 'weight', None)
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
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
# CUDA Kernel: Fused GEMM + Bias + ReLU
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// 16x16 tile size is chosen to be warp-friendly and fit in shared memory
#define TILE_DIM 16

__global__ void fused_gemm_relu_kernel(
    const float* __restrict__ A,   // M x K
    const float* __restrict__ B,   // K x N (transposed weight)
    const float* __restrict__ bias,// N
    float*       __restrict__ C,   // M x N
    int M, int N, int K)
{
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float acc = 0.0f;

    // K dimension loop (tiled)
    for (int k = 0; k < K; k += TILE_DIM) {
        // Load tile A
        if (row < M && (k + tx) < K)
            As[ty][tx] = A[row * K + k + tx];
        else
            As[ty][tx] = 0.0f;

        // Load tile B
        if ((k + ty) < K && col < N)
            Bs[ty][tx] = B[(k + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute dot product for this tile
        #pragma unroll
        for (int i = 0; i < TILE_DIM; ++i) {
            acc += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        float val = acc + bias[col];
        C[row * N + col] = fmaxf(0.0f, val);
    }
}

void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor bias, torch::Tensor C) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    fused_gemm_relu_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        bias.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
}
"""

# ----------------------------------------------------------------------
# C++ Bindings
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor bias, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused GEMM + Bias + ReLU");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op_module',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# ----------------------------------------------------------------------
# Python Entry Point
# ----------------------------------------------------------------------
def functional_model(x, *, gemm_weight, bias):
    # Weight must be N x K, transform to K x N for kernel
    # gemm_weight is [out_features, in_features]
    weight_T = gemm_weight.t().contiguous()
    
    batch_size = x.size(0)
    out_features = gemm_weight.size(0)
    
    out = torch.empty((batch_size, out_features), device=x.device, dtype=torch.float32)
    
    # Kernel handles matrix multiplication, bias addition, and ReLU
    fused_ext.fused_op(x.contiguous(), weight_T, bias.contiguous(), out)
    
    return out

def get_init_inputs():
    return [8192, 8192, (8192,)]

def get_inputs():
    return [torch.rand(1024, 8192, device='cuda')]
