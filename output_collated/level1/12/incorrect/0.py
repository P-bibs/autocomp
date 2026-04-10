# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_102701/code_7.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 256

__global__ void fused_tiled_multiply_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M
) {
    __shared__ float s_A[TILE_SIZE];
    
    int tid = threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + tid;
    
    // Load tile of A into shared memory
    if (row < N) {
        s_A[tid] = A[row];
    }
    __syncthreads();
    
    // Each thread processes one row of B for current tile
    if (row < N) {
        float a_val = s_A[tid];
        int b_row_offset = row * M;
        
        // Process columns in steps of gridDim.x * blockDim.x
        for (int col = blockIdx.x * blockDim.x + tid; col < M; col += gridDim.x * blockDim.x) {
            C[b_row_offset + col] = a_val * B[b_row_offset + col];
        }
    }
}

void fused_op_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int N, int M) {
    const int threads_per_block = 256;
    const int blocks_x = (M + threads_per_block - 1) / threads_per_block;
    const int blocks_y = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    dim3 blocks(blocks_x, blocks_y);
    dim3 threads(threads_per_block);
    
    fused_tiled_multiply_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int N, int M);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Tiled fused unsqueeze and multiply operation");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    N, M = B.shape
    
    # Allocate output tensor
    C = torch.empty(N, M, dtype=torch.float32, device='cuda')
    
    # Ensure inputs are on GPU and correct dtype
    A_gpu = A.to(torch.float32).cuda().contiguous()
    B_gpu = B.to(torch.float32).cuda().contiguous()
    
    # Call custom CUDA kernel
    fused_ext.fused_op(A_gpu, B_gpu, C, N, M)
    
    return C
