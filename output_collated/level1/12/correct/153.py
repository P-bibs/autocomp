# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110950/code_26.py
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

# ------------------------------------------------------------------
#  CUDA kernel – leverages shared memory for tile-based broadcasted 
#  multiplication. Each row of A is broadcast across M columns of B.
# ------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_THREADS 128
#define ELEMENTS_PER_THREAD 4
#define TILE_SIZE (BLOCK_THREADS * ELEMENTS_PER_THREAD)

__global__ void broadcast_mul_shared_kernel(
    const float* __restrict__ A,          // (N,)
    const float* __restrict__ B,          // (N, M)
    float*       __restrict__ out,        // (N, M)
    const int N,
    const int M)
{
    const int row = blockIdx.y;
    if (row >= N) return;

    // Load the broadcast scalar into register
    const float a_val = A[row];

    // Shared memory: 128 threads * 4 floats = 512 floats (2KB) per block
    __shared__ float shB[TILE_SIZE];

    // Pointer offsets for the current row
    const float* row_B = B + (row * M);
    float* row_out = out + (row * M);

    for (int col = blockIdx.x * TILE_SIZE; col < M; col += gridDim.x * TILE_SIZE) {
        // 1) Coalesced load from global memory into shared memory
        int tid = threadIdx.x;
        int lane_idx = tid * ELEMENTS_PER_THREAD;
        
        if (col + lane_idx + 3 < M) {
            float4 b_vec = reinterpret_cast<const float4*>(&row_B[col + lane_idx])[0];
            reinterpret_cast<float4*>(&shB[lane_idx])[0] = b_vec;
        } else {
            // Handle boundary conditions for matrix width M not divisible by TILE_SIZE
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                if (col + lane_idx + i < M)
                    shB[lane_idx + i] = row_B[col + lane_idx + i];
            }
        }

        __syncthreads();

        // 2) Compute and write back
        if (col + lane_idx + 3 < M) {
            float4 s_vec = reinterpret_cast<float4*>(&shB[lane_idx])[0];
            float4 out_vec;
            out_vec.x = a_val * s_vec.x;
            out_vec.y = a_val * s_vec.y;
            out_vec.z = a_val * s_vec.z;
            out_vec.w = a_val * s_vec.w;
            reinterpret_cast<float4*>(&row_out[col + lane_idx])[0] = out_vec;
        } else {
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                if (col + lane_idx + i < M)
                    row_out[col + lane_idx + i] = a_val * shB[lane_idx + i];
            }
        }
        __syncthreads();
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& out) {
    const int N = B.size(0);
    const int M = B.size(1);
    
    // Grid dim: x-blocks cover width, y-blocks cover height (rows)
    dim3 block(BLOCK_THREADS);
    dim3 grid((M + TILE_SIZE - 1) / TILE_SIZE, N);
    
    broadcast_mul_shared_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        M
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward, "Vectorized broadcast multiplication with shared memory");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='broadcast_mul_shared_ext',
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
