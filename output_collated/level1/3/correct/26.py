# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_092316/code_4.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['A', 'B']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

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

# We use a 32x32 tile with padding to avoid shared memory bank conflicts
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define PAD_DIM 33

__global__ void batch_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int m, int n, int k
) {
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    __shared__ float sA[TILE_DIM][PAD_DIM];
    __shared__ float sB[TILE_DIM][PAD_DIM];

    float sum = 0.0f;
    const float* A_ptr = A + batch_idx * m * k;
    const float* B_ptr = B + batch_idx * k * n;

    for (int t = 0; t < (k + TILE_DIM - 1) / TILE_DIM; ++t) {
        int t_idx = t * TILE_DIM;
        
        // Coalesced loading into shared memory
        if (row < m && (t_idx + threadIdx.x) < k)
            sA[threadIdx.y][threadIdx.x] = A_ptr[row * k + (t_idx + threadIdx.x)];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if ((t_idx + threadIdx.y) < k && col < n)
            sB[threadIdx.y][threadIdx.x] = B_ptr[(t_idx + threadIdx.y) * n + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_DIM; ++i) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        C[batch_idx * m * n + row * n + col] = sum;
    }
}

void batch_matmul_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    int batch_size, int m, int n, int k
) {
    dim3 block_dim(TILE_DIM, TILE_DIM);
    dim3 grid_dim((n + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM, batch_size);
    
    batch_matmul_kernel<<<grid_dim, block_dim>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), 
        m, n, k
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void batch_matmul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int batch_size, int m, int n, int k);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_matmul", &batch_matmul_forward, "Batch Matmul");
}
"""

# Compile extension
fused_ext = load_inline(
    name='batch_matmul_ext', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(A, B):
    batch_size = A.size(0)
    m, k, n = A.size(1), A.size(2), B.size(2)
    C = torch.empty(batch_size, m, n, device=A.device, dtype=A.dtype)
    fused_ext.batch_matmul(A, B, C, batch_size, m, n, k)
    return C
