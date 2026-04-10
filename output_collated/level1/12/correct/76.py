# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_103853/code_25.py
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

# Optimization: Using shared memory to cache row-vector A
# TILE_N = 32, TILE_M = 128 (processes 32 float4s per row chunk)
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_N 32
#define TILE_M 128

__global__ void broadcast_mul_shared_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    extern __shared__ float shmem[];
    float* shA = shmem; 

    int block_row = blockIdx.y * TILE_N;
    int block_col = blockIdx.x * TILE_M;

    // Load tile of A into shared memory
    // TILE_N threads load 1 element each
    if (threadIdx.y < TILE_N && (block_row + threadIdx.y) < N) {
        if (threadIdx.x == 0) {
            shA[threadIdx.y] = A[block_row + threadIdx.y];
        }
    }
    __syncthreads();

    // Each thread processes 4 elements (1 float4)
    int tid_m = (threadIdx.x * 4);
    int row = block_row + threadIdx.y;
    int col = block_col + tid_m;

    if (row < N && col + 3 < M) {
        float a_val = shA[threadIdx.y];
        float4 b_vec = reinterpret_cast<const float4*>(&B[row * M + col])[0];
        
        float4 out_vec;
        out_vec.x = a_val * b_vec.x;
        out_vec.y = a_val * b_vec.y;
        out_vec.z = a_val * b_vec.z;
        out_vec.w = a_val * b_vec.w;
        
        reinterpret_cast<float4*>(&output[row * M + col])[0] = out_vec;
    } else if (row < N) {
        // Scalar fallback for boundary
        for (int i = 0; i < 4; ++i) {
            if (col + i < M) {
                output[row * M + col + i] = A[row] * B[row * M + col + i];
            }
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    
    dim3 threads(TILE_M / 4, TILE_N);
    dim3 blocks((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    
    // Shared memory required for shA
    size_t shmem_size = TILE_N * sizeof(float);
    
    broadcast_mul_shared_kernel<<<blocks, threads, shmem_size>>>(
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
    m.def("broadcast_mul", &broadcast_mul_forward, "Vectorized Tiled broadcast multiplication");
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

N, M = 4096, 4096

def get_inputs():
    A = torch.rand(N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]
