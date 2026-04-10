# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110950/code_16.py
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

# The kernel is optimized to use grid-stride loops, 128-bit loads (float4),
# and explicit register usage to minimize memory round-trips.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_forward_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements per iteration to fully utilize float4 memory bandwidth
    for (int i = tid * 4; i < N * M; i += stride * 4) {
        // Check if we are at the end of a row. If we cross a row boundary, 
        // the float4 load would be invalid for the specific operation (A[row] * B[i]).
        // We only use the vectorized path if the 4 elements belong to the same row.
        int row = i / M;
        bool cross_boundary = ((i + 3) / M != row);
        
        if (!cross_boundary && (i + 3 < N * M)) {
            float4 b_vec = reinterpret_cast<const float4*>(&B[i])[0];
            float a_val = A[row];
            
            float4 c_vec;
            c_vec.x = a_val * b_vec.x;
            c_vec.y = a_val * b_vec.y;
            c_vec.z = a_val * b_vec.z;
            c_vec.w = a_val * b_vec.w;
            
            reinterpret_cast<float4*>(&C[i])[0] = c_vec;
        } else {
            // Scalar fallback for row boundaries or end of buffer
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int idx = i + k;
                if (idx < N * M) {
                    C[idx] = A[idx / M] * B[idx];
                }
            }
        }
    }
}

void fused_op_forward(int blocks, int threads, const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int N, int M) {
    fused_op_forward_kernel<<<blocks, threads>>>(
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
void fused_op_forward(int blocks, int threads, const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int N, int M);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Vectorized fused scalar-matrix multiply");
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
    C = torch.empty(N, M, dtype=torch.float32, device='cuda')
    
    A_contig = A.contiguous()
    B_contig = B.contiguous()
    
    # Heuristic for block/grid sizing
    threads = 256
    # Calculate grid size to ensure occupancy while respecting 65535 limit
    total_chunks = (N * M + 3) // 4
    blocks = min(65535, (total_chunks + threads - 1) // threads)
    
    fused_ext.fused_op(blocks, threads, A_contig, B_contig, C, N, M)
    return C
