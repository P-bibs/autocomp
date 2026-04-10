# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112116/code_24.py
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

# ----------------------------------------------------------------------
# CUDA kernel: One block per row, each thread processes multiple columns
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <int VEC>
__global__ void fused_op_forward_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int N,
    int M)
{
    // Map block to row. If N is very large, the launcher handles multi-row handling.
    // Here we use the block's Y-coordinate to handle N > 65535
    int row = blockIdx.x + blockIdx.y * 65535;
    if (row >= N) return;

    float a_val = A[row];
    int col_start = threadIdx.x * VEC;
    int stride = blockDim.x * VEC;

    const float* B_row = B + static_cast<long long>(row) * M;
    float*       C_row = C + static_cast<long long>(row) * M;

    for (int col = col_start; col < M; col += stride) {
        if (col + VEC - 1 < M) {
            float4 b_vec = reinterpret_cast<const float4*>(B_row + col)[0];
            float4 c_vec;
            c_vec.x = a_val * b_vec.x;
            c_vec.y = a_val * b_vec.y;
            c_vec.z = a_val * b_vec.z;
            c_vec.w = a_val * b_vec.w;
            reinterpret_cast<float4*>(C_row + col)[0] = c_vec;
        } else {
            // Tail handling
            for (int k = 0; k < VEC; ++k) {
                int idx = col + k;
                if (idx < M) {
                    C_row[idx] = a_val * B_row[idx];
                }
            }
        }
    }
}

void fused_op_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int N, int M) {
    const int threads = 256;
    
    // We want to process each row. If N > 65535, we use Y-dimension for blocks.
    dim3 grid;
    if (N <= 65535) {
        grid = dim3(N, 1, 1);
    } else {
        grid = dim3(65535, (N + 65534) / 65535, 1);
    }

    fused_op_forward_kernel<4><<<grid, threads>>>(
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
    m.def("fused_op", &fused_op_forward, "Vectorized fused unsqueeze-multiply");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False
)

def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized implementation:
    - Minimizes global memory reads by caching A[row] in a register.
    - Achieves perfect coalesced global memory access for B and C.
    - Minimizes kernel launch overhead by sizing blocks to rows.
    """
    if A.dim() != 1 or B.dim() != 2:
        raise ValueError("Dimensions mismatch.")
    
    N, M = B.shape
    # Ensure inputs are contiguous
    A_contig = A.contiguous().cuda()
    B_contig = B.contiguous().cuda()
    C = torch.empty(N, M, dtype=torch.float32, device='cuda')
    
    fused_ext.fused_op(A_contig, B_contig, C, N, M)
    return C
