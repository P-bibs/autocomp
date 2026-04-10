# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105940/code_12.py
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

// Optimized kernel using shared memory for A values
// Each block processes multiple rows, caching A[row] in shared memory
__global__ void fused_op_forward_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M
) {
    // Shared memory for caching A values
    // Size: blockDim.x elements (one per thread in block)
    extern __shared__ float shared_A[];
    
    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x;
    
    // Each thread loads its corresponding A value (if valid)
    // This is coalesced because threads access consecutive memory locations
    int row = blockStart + tid;
    if (row < N) {
        shared_A[tid] = A[row];
    }
    __syncthreads();
    
    // Each thread processes all elements in its assigned row
    if (row < N) {
        float a_val = shared_A[tid];
        int row_offset = row * M;
        
        // Vectorized processing: 4 elements at a time
        int vecs_per_row = M / 4;
        int remainder = M % 4;
        
        // Process vectorized chunks
        for (int i = 0; i < vecs_per_row; i++) {
            int col = i * 4;
            float4 b_vec = reinterpret_cast<const float4*>(&B[row_offset + col])[0];
            float4 c_vec;
            c_vec.x = a_val * b_vec.x;
            c_vec.y = a_val * b_vec.y;
            c_vec.z = a_val * b_vec.z;
            c_vec.w = a_val * b_vec.w;
            reinterpret_cast<float4*>(&C[row_offset + col])[0] = c_vec;
        }
        
        // Handle remainder elements
        for (int j = 0; j < remainder; j++) {
            int col = vecs_per_row * 4 + j;
            C[row_offset + col] = a_val * B[row_offset + col];
        }
    }
}

void fused_op_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int N, int M) {
    const int threads = 256;
    
    // Grid: ceil(N / threads) blocks
    // Each block handles up to 'threads' rows
    int num_blocks = (N + threads - 1) / threads;
    
    // Shared memory: threads * sizeof(float) for A cache
    int shared_mem_size = threads * sizeof(float);
    
    fused_op_forward_kernel<<<num_blocks, threads, shared_mem_size>>>(
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
    m.def("fused_op", &fused_op_forward, "Vectorized fused unsqueeze-multiply with shared memory");
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
    # Kernel assumes contiguous inputs for optimal performance
    A_contig = A.contiguous().cuda()
    B_contig = B.contiguous().cuda()
    fused_ext.fused_op(A_contig, B_contig, C, N, M)
    return C
