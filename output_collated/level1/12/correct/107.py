# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_104927/code_31.py
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
# 1. CUDA kernel and Host wrapper
# ------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Grid-stride kernel that computes C[row, col] = A[row] * B[row, col]
// This layout ensures coalesced memory access to matrix B.
__global__ void mul_kernel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float*       __restrict__ C,
                           const int N,
                           const int M)
{
    const int total = N * M;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        int row = i / M;
        int col = i % M;
        // A[row] is broadcast across threads in a warp
        // B[i] is accessed sequentially, leading to coalesced global memory loads
        C[i] = A[row] * B[i];
    }
}

void mul_cuda(const torch::Tensor& A,
              const torch::Tensor& B,
              torch::Tensor& C)
{
    const int N = A.size(0);
    const int M = B.size(1);
    const int total = N * M;

    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    mul_kernel<<<grid_size, block_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M
    );
}
"""

# ------------------------------------------------------------
# 2. C++ bindings via PyBind11
# ------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void mul_cuda(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mul", &mul_cuda, "Fused A[row] * B[row, col] kernel");
}
"""

# ------------------------------------------------------------
# 3. Compile the inline extension
# ------------------------------------------------------------
fused_ext = load_inline(
    name='fused_mul',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ------------------------------------------------------------
# 4. Optimized functional_model
# ------------------------------------------------------------
def functional_model(A, B):
    """
    Optimized replacement for A.unsqueeze(1) * B using a fused CUDA kernel.
    """
    # Ensure inputs are on GPU and contiguous
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    B = B.contiguous()
    A = A.contiguous()

    N = A.shape[0]
    M = B.shape[1]
    
    # Pre-allocate output tensor
    C = torch.empty((N, M), dtype=A.dtype, device=A.device)
    
    # Launch CUDA kernel
    fused_ext.mul(A, B, C)
    
    return C

# ------------------------------------------------------------
# 5. Requirement Boilerplate
# ------------------------------------------------------------
M = 4096
N = 4096

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]
