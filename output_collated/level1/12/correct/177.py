# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112116/code_20.py
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
    // Map grid to 2D: blockIdx.y handles rows, blockIdx.x handles columns
    // We use a grid-stride loop for the rows to handle N > 65535
    for (int row = blockIdx.y; row < N; row += gridDim.y) {
        float a_val = __ldg(&A[row]); // Use read-only cache for A
        
        int col_step = blockDim.x * 4;
        for (int col = blockIdx.x * col_step + threadIdx.x * 4; col < M; col += blockDim.x * gridDim.x * 4) {
            if (col + 3 < M) {
                float4 b_vec = reinterpret_cast<const float4*>(&B[row * M + col])[0];
                float4 c_vec;
                c_vec.x = a_val * b_vec.x;
                c_vec.y = a_val * b_vec.y;
                c_vec.z = a_val * b_vec.z;
                c_vec.w = a_val * b_vec.w;
                reinterpret_cast<float4*>(&C[row * M + col])[0] = c_vec;
            } else {
                // Tail handling for M not aligned to 4
                for (int i = 0; i < 4; ++i) {
                    if (col + i < M) {
                        C[row * M + col + i] = a_val * B[row * M + col + i];
                    }
                }
            }
        }
    }
}

void fused_op_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int N, int M) {
    const int threads = 256;
    // Limit grid.y for compatibility and performance
    int grid_y = std::min(N, 65535);
    int grid_x = std::min((M / 4 + threads - 1) / threads, 64);
    
    dim3 grid(grid_x, grid_y);
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
    m.def("fused_op", &fused_op_forward, "Optimized broadcasting multiply");
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
    C = torch.empty_like(B)
    # Ensure inputs are contiguous to satisfy the memory access pattern
    fused_ext.fused_op(
        A.contiguous(), 
        B.contiguous(), 
        C, 
        N, 
        M
    )
    return C
