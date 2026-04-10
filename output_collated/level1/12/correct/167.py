# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112116/code_8.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['A', 'B']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
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

# Optimized CUDA kernel with reduced global memory accesses
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <int VEC>
__global__ void fused_op_forward_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M
) {
    // One block processes one row
    int row = blockIdx.x;
    if (row >= N) return;

    // Load scalar once per block (reduces global memory accesses)
    float a_val = A[row];

    // Each thread processes VEC elements at a time
    int col_start = threadIdx.x * VEC;
    int stride = blockDim.x * VEC;

    // Row pointers for B and C
    const float* B_row = B + static_cast<long long>(row) * M;
    float* C_row = C + static_cast<long long>(row) * M;

    // Process columns with grid-stride loop
    for (int col = col_start; col < M; col += stride) {
        if (col + VEC - 1 < M) {
            // Vectorized path
            float4 b_vec = reinterpret_cast<const float4*>(&B_row[col])[0];
            float4 c_vec;
            c_vec.x = a_val * b_vec.x;
            c_vec.y = a_val * b_vec.y;
            c_vec.z = a_val * b_vec.z;
            c_vec.w = a_val * b_vec.w;
            reinterpret_cast<float4*>(&C_row[col])[0] = c_vec;
        } else {
            // Handle tail elements
            #pragma unroll
            for (int k = 0; k < VEC; ++k) {
                int idx = col + k;
                if (idx < M) {
                    C_row[idx] = a_val * B_row[idx];
                }
            }
        }
    }
}

void fused_op_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int N, int M) {
    const int threads = 256;
    const int blocks = N;

    dim3 grid;
    if (blocks <= 65535) {
        grid = dim3(blocks, 1, 1);
    } else {
        int grid_x = 65535;
        int grid_y = (blocks + grid_x - 1) / grid_x;
        grid = dim3(grid_x, grid_y, 1);
    }

    fused_op_forward_kernel<4><<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in fused_op_forward: %s\n", cudaGetErrorString(err));
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int N, int M);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Optimized fused unsqueeze-multiply");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Perform C[i, j] = A[i] * B[i, j] for a (N,) vector A and an (N, M) matrix B.
    Uses optimized CUDA kernel with reduced global memory accesses.
    """
    if A.dim() != 1:
        raise ValueError("A must be a 1-D tensor")
    if B.dim() != 2:
        raise ValueError("B must be a 2-D tensor")
    N, M = B.shape
    if A.shape[0] != N:
        raise ValueError("A length must match B's first dimension")

    # Ensure contiguous, GPU tensors
    A_contig = A.contiguous().cuda()
    B_contig = B.contiguous().cuda()
    
    # Allocate output
    C = torch.empty_like(B_contig)

    # Launch optimized kernel
    fused_ext.fused_op(A_contig, B_contig, C, N, M)

    return C
