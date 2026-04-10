# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_104927/code_24.py
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
#include <vector_types.h>

extern __shared__ float a_tile[];

__global__ void fused_op_forward_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float a_val = 0.0f;
    
    // Load row-specific scalar A[row] once into shared memory
    if (row < N) {
        a_tile[threadIdx.x] = A[row];
    }
    __syncthreads();
    
    if (row < N) {
        a_val = a_tile[threadIdx.x];
        
        // Process columns in strides
        for (int col = 0; col < M; col += blockDim.x * 4) {
            int current_col = col + threadIdx.x * 4;
            
            // Check if we can perform a vectorized float4 load
            if (current_col + 3 < M) {
                float4 b_vec = reinterpret_cast<const float4*>(&B[row * M + current_col])[0];
                float4 c_vec;
                c_vec.x = a_val * b_vec.x;
                c_vec.y = a_val * b_vec.y;
                c_vec.z = a_val * b_vec.z;
                c_vec.w = a_val * b_vec.w;
                reinterpret_cast<float4*>(&C[row * M + current_col])[0] = c_vec;
            } else {
                // Scalar fallback for tail elements
                for (int k = 0; k < 4; ++k) {
                    int c = current_col + k;
                    if (c < M) {
                        C[row * M + c] = a_val * B[row * M + c];
                    }
                }
            }
        }
    }
}

void fused_op_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int N, int M) {
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    const size_t shared_mem = threads * sizeof(float);

    fused_op_forward_kernel<<<blocks, threads, shared_mem>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int N, int M);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Optimized vectorized fused unsqueeze-multiply");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Optimized implementation of element-wise multiplication A[row] * B[row, col].
    Uses shared memory to cache the row scalar of A, minimizing global memory traffic.
    """
    N, M = B.shape
    C = torch.empty(N, M, dtype=torch.float32, device='cuda')
    
    # Ensure inputs are contiguous to satisfy kernel's memory indexing patterns
    A_contig = A.contiguous()
    B_contig = B.contiguous()
    
    fused_ext.fused_op(A_contig, B_contig, C, N, M)
    return C
