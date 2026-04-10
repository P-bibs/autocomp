# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105940/code_24.py
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

# -------------------------------------------------------------------------
# CUDA kernel – uses shared memory to cache A for reduced global memory traffic
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" __global__
void fused_op_forward_kernel(
    const float* __restrict__ A,          // [N]
    const float* __restrict__ B,          // [N, M]
    float*       __restrict__ C,          // [N, M]
    int N,
    int M
) {
    // Shared memory for one block of the broadcast vector A
    // Size is configured at launch to be blockDim.x * sizeof(float)
    extern __shared__ float shA[];

    int tid = threadIdx.x;
    int row = blockIdx.x * blockDim.x + tid;

    // 1. Cooperative load: Each thread in the block loads its assigned 
    // row's scalar from A into shared memory.
    if (row < N) {
        shA[tid] = A[row];
    } else {
        shA[tid] = 0.0f; // Safety padding
    }
    __syncthreads();

    // 2. Vectorized processing: Threads iterate over their assigned row
    // using the cached value from shA.
    // Each thread processes 4 elements at a time.
    int idx = (blockIdx.x * blockDim.x + tid) * 4;
    int total_elements = N * M;

    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x * 4) {
        int current_row = i / M;
        // If current thread has spanned beyond its assigned row, break.
        if (current_row >= N + (blockIdx.x * blockDim.x + tid) - (blockIdx.x * blockDim.x + tid)) {
             // In this logic, each thread is responsible for one row. 
             // Once the row is exhausted, the thread is done.
             if (current_row != (blockIdx.x * blockDim.x + tid)) break;
        }

        float a_val = shA[tid];

        if (i + 3 < (current_row + 1) * M) {
            float4 b_vec = reinterpret_cast<const float4*>(&B[i])[0];
            float4 c_vec;
            c_vec.x = a_val * b_vec.x;
            c_vec.y = a_val * b_vec.y;
            c_vec.z = a_val * b_vec.z;
            c_vec.w = a_val * b_vec.w;
            reinterpret_cast<float4*>(&C[i])[0] = c_vec;
        } else {
            // Tail handling
            for (int j = 0; j < 4; ++j) {
                int flat = i + j;
                if (flat < total_elements && (flat / M) == current_row) {
                    C[flat] = a_val * B[flat];
                }
            }
        }
    }
}

void fused_op_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int N, int M) {
    const int threads = 256;
    int num_blocks = (N + threads - 1) / threads;
    
    // Launch with dynamic shared memory for shA
    size_t shmem_size = threads * sizeof(float);
    
    fused_op_forward_kernel<<<num_blocks, threads, shmem_size>>>(
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
    m.def("fused_op", &fused_op_forward, "Vectorized fused broadcast-multiply with shared memory");
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
    """Computes A.unsqueeze(1) * B using a fused CUDA kernel."""
    N, M = B.shape
    C = torch.empty(N, M, dtype=torch.float32, device='cuda')
    
    # Ensure inputs are contiguous
    A_contig = A.contiguous().cuda()
    B_contig = B.contiguous().cuda()
    
    fused_ext.fused_op(A_contig, B_contig, C, N, M)
    return C
