# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_101548/code_3.py
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

# ------------------------------------------------------------
# 1.  Load the custom CUDA kernel via torch.utils.cpp_extension
# ------------------------------------------------------------

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__global__ void fused_op_forward_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N,
    const int M
) {
    // Shared memory for caching A values per row in the block
    __shared__ float sA[BLOCK_SIZE_Y];

    // Calculate global thread indices
    int col = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    int row = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;

    // Check bounds
    bool valid = (row < N && col < M);

    // Cooperative loading of A into shared memory
    if (valid && threadIdx.x == 0) {
        sA[threadIdx.y] = A[row];
    }
    __syncthreads();

    // Perform element-wise multiplication using cached A value
    if (valid) {
        float a = sA[threadIdx.y];
        float b = B[row * M + col];
        C[row * M + col] = a * b;
    }
}

void fused_op_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C
) {
    int N = A.size(0);
    int M = B.size(1);

    dim3 blocks((M + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                (N + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    
    fused_op_forward_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M
    );
    
    // No need for explicit synchronization as PyTorch handles it
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function in the .cu file
void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Optimized A.unsqueeze(1) * B with shared memory caching");
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

# ------------------------------------------------------------
# 2.  functional_model – the function that will be evaluated
# ------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Returns A.unsqueeze(1) * B using a custom CUDA kernel that caches
    the per-row scaling factor in shared memory.
    """
    # Ensure tensors are on GPU
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()

    # Allocate output tensor
    C = torch.empty_like(B)

    # Launch the optimized kernel
    fused_ext.fused_op(A, B, C)

    return C

# ------------------------------------------------------------
# 3.  Helper functions required by the harness
# ------------------------------------------------------------
M = 4096
N = 4096

def get_init_inputs():
    """No special initialization inputs are needed."""
    return []

def get_inputs():
    """Produce the same random inputs as the original benchmark."""
    A = torch.rand(N)          # shape (N,)
    B = torch.rand(N, M)       # shape (N,M)
    return [A, B]

# ------------------------------------------------------------
# 4.  Quick sanity-check (can be removed in production)
# ------------------------------------------------------------
if __name__ == "__main__":
    A, B = get_inputs()
    C = functional_model(A, B)
    # Reference computed with pure PyTorch
    C_ref = A.unsqueeze(1) * B
    print("Max absolute difference:", (C - C_ref).abs().max().item())
