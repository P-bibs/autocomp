# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_102701/code_19.py
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

# The operation is C[i, j] = A[i] * B[i, j]. 
# Since B is stored row-major, this is perfectly coalesced if 
# threads process elements in the same row together.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_unsqueeze_multiply_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * M;

    // Grid-stride loop ensures efficiency regardless of input size
    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        int row = i / M;
        // B[i] and C[i] are accessed strictly coalesced
        C[i] = A[row] * B[i];
    }
}

void fused_unsqueeze_multiply_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C
) {
    const int N = A.size(0);
    const int M = B.size(1);
    const int total_elements = N * M;
    
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    fused_unsqueeze_multiply_kernel<<<blocks, threads>>>(
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

void fused_unsqueeze_multiply_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_unsqueeze_multiply", &fused_unsqueeze_multiply_forward, "Fused elementwise multiply");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_unsqueeze_multiply',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    # A is (N,), B is (N, M)
    N = A.shape[0]
    M = B.shape[1]
    
    C = torch.empty((N, M), dtype=torch.float32, device='cuda')
    
    # Ensure inputs are on GPU
    A_gpu = A.to(device='cuda', dtype=torch.float32)
    B_gpu = B.to(device='cuda', dtype=torch.float32)
    
    fused_ext.fused_unsqueeze_multiply(A_gpu, B_gpu, C)
    
    return C
