# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_104927/code_20.py
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

__global__ void fused_op_forward_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* row_B = B + row * M;
    float* row_C = C + row * M;
    float a_val = A[row];

    int M_vec = M / 4;
    int col = threadIdx.x;

    // Vectorized path: Process 4 elements at a time
    for (int i = col; i < M_vec; i += blockDim.x) {
        float4 b_vec = reinterpret_cast<const float4*>(&row_B[i * 4])[0];
        float4 c_vec = {a_val * b_vec.x, a_val * b_vec.y, a_val * b_vec.z, a_val * b_vec.w};
        reinterpret_cast<float4*>(&row_C[i * 4])[0] = c_vec;
    }

    // Tail path: Process remaining elements
    for (int i = M_vec * 4 + col; i < M; i += blockDim.x) {
        row_C[i] = a_val * row_B[i];
    }
}

void fused_op_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int N, int M) {
    // 128 threads is a sweet spot for RTX 2080Ti occupancy
    int threads = 128;
    // Map each row to a block. 
    dim3 grid(N);
    
    fused_op_forward_kernel<<<grid, threads>>>(
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
    m.def("fused_op", &fused_op_forward, "Optimized float4 vectorization kernel");
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
    """
    Computes C = A.unsqueeze(1) * B element-wise.
    Optimized to use float4 vectorization and thread-coalesced memory access.
    """
    A = A.view(-1)
    N, M = B.shape
    
    # Pre-allocate output on CUDA
    C = torch.empty_like(B)
    
    # Ensure inputs are contiguous; if not, contiguous() handles it
    A_contig = A.contiguous().cuda()
    B_contig = B.contiguous().cuda()
    
    fused_ext.fused_op(A_contig, B_contig, C, N, M)
    return C
