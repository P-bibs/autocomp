# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105940/code_20.py
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
    // Optimization: Use a 2D block/grid strategy to eliminate integer division (i / M)
    // blockIdx.y represents the row index
    // blockIdx.x and threadIdx.x cover the columns (M)
    int row = blockIdx.y;
    if (row >= N) return;

    const float a_val = A[row];
    const float* __restrict__ B_row = B + (row * M);
    float* __restrict__ C_row = C + (row * M);

    for (int col = blockIdx.x * blockDim.x + threadIdx.x; 
         col < M; 
         col += blockDim.x * gridDim.x) {
        C_row[col] = a_val * B_row[col];
    }
}

void fused_op_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int N, int M) {
    // 256 threads per block is usually sweet spot for 2080Ti (Turing)
    const int threads_x = 256;
    
    // We want as many blocks as possible in X up to hardware limit, 
    // and one block per row in Y.
    dim3 threads(threads_x, 1);
    dim3 blocks((M + threads_x - 1) / threads_x, N);
    
    // Constraint: Max grid dimension Y is 65535 on older architectures, 
    // but on 2080Ti (Compute Cap 7.5), grid Y can be 2^31-1.
    // We clamp to 65535 for broader compatibility if N is large.
    if (blocks.y > 65535) {
        // Fallback for extremely large N: loop inside kernel or use 1D grid
        blocks.y = 65535;
    }

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
void fused_op_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int N, int M);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Optimized fused row-wise multiply");
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
    device = B.device
    C = torch.empty(N, M, dtype=torch.float32, device=device)
    
    # Ensure contiguous inputs for coalesced memory access
    A_contig = A.contiguous().cuda()
    B_contig = B.contiguous().cuda()
    
    fused_ext.fused_op(A_contig, B_contig, C, N, M)
    return C
