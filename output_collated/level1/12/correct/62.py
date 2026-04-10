# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_103853/code_9.py
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

#define TILE_N 32
#define TILE_M 128

__global__ void broadcast_mul_vectorized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    // Shared memory for tile of A and tile of B
    extern __shared__ float shmem[];
    float* shA = shmem;                      // TILE_N elements
    float* shB = shmem + TILE_N;            // TILE_N * TILE_M elements

    // Block and thread indices
    int blockCol = blockIdx.x * TILE_M;
    int blockRow = blockIdx.y * TILE_N;
    int tx = threadIdx.x;   // 0 to TILE_M/4 - 1 (since each thread handles 4 elements)
    int ty = threadIdx.y;   // 0 to TILE_N - 1

    // Load tile of A into shared memory
    if (ty < TILE_N && (blockRow + ty) < N) {
        shA[ty] = A[blockRow + ty];
    } else if (ty < TILE_N) {
        shA[ty] = 0.0f;
    }
    
    // Load tile of B into shared memory (each thread loads 4 consecutive elements)
    int col = tx * 4;
    if (col < TILE_M && (blockRow + ty) < N) {
        int globalIdx = (blockRow + ty) * M + blockCol + col;
        if (globalIdx + 3 < N * M) {
            // Safe to load full float4
            reinterpret_cast<float4*>(shB)[ty * (TILE_M / 4) + (col / 4)] = 
                reinterpret_cast<const float4*>(B)[globalIdx / 4];
        } else {
            // Handle remainder elements
            for (int i = 0; i < 4 && (blockCol + col + i) < M; ++i) {
                shB[ty * TILE_M + col + i] = B[globalIdx + i];
            }
        }
    }

    __syncthreads();

    // Compute results using shared memory
    if ((blockRow + ty) < N) {
        float a_val = shA[ty];
        int out_row = blockRow + ty;
        
        if (col + 3 < TILE_M && (blockCol + col + 3) < M) {
            // Process 4 elements at once
            float4 b_vec = reinterpret_cast<float4*>(shB)[ty * (TILE_M / 4) + (col / 4)];
            float4 out_vec;
            out_vec.x = a_val * b_vec.x;
            out_vec.y = a_val * b_vec.y;
            out_vec.z = a_val * b_vec.z;
            out_vec.w = a_val * b_vec.w;
            
            int out_idx = out_row * M + blockCol + col;
            reinterpret_cast<float4*>(output)[out_idx / 4] = out_vec;
        } else {
            // Handle remainder columns
            for (int i = 0; i < 4 && (blockCol + col + i) < M; ++i) {
                int out_idx = out_row * M + blockCol + col + i;
                output[out_idx] = a_val * shB[ty * TILE_M + col + i];
            }
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    
    // Grid and block dimensions
    dim3 gridDim((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    dim3 blockDim(TILE_M / 4, TILE_N);
    
    // Shared memory size: TILE_N floats for A + TILE_N * TILE_M floats for B
    const int shmem_size = (TILE_N + TILE_N * TILE_M) * sizeof(float);
    
    broadcast_mul_vectorized_kernel<<<gridDim, blockDim, shmem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        M
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward, "Vectorized broadcast multiplication with shared memory");
}
"""

fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output

M = 4096
N = 4096

def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]
