# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110950/code_0.py
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
#include <vector_types.h>

__global__ void fused_op_forward_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    int total_elements = N * M;

    for (int base_idx = tid * 4; base_idx < total_elements; base_idx += total_threads * 4) {
        int row = base_idx / M;
        if (row >= N) break;
        float a_val = A[row];

        // Check if we can do a vectorized load/store
        if (base_idx + 3 < total_elements && (base_idx + 3) / M == row) {
            float4 b_vec = reinterpret_cast<const float4*>(&B[base_idx])[0];
            float4 c_vec;
            c_vec.x = a_val * b_vec.x;
            c_vec.y = a_val * b_vec.y;
            c_vec.z = a_val * b_vec.z;
            c_vec.w = a_val * b_vec.w;
            reinterpret_cast<float4*>(&C[base_idx])[0] = c_vec;
        } else {
            // Handle scalar elements (including tails)
            for (int j = 0; j < 4 && base_idx + j < total_elements; ++j) {
                int curr_row = (base_idx + j) / M;
                if (curr_row != row) break;
                C[base_idx + j] = a_val * B[base_idx + j];
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
    m.def("fused_op", &fused_op_forward, "Optimized fused unsqueeze-multiply with coalesced memory");
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
    A_contig = A.contiguous().cuda()
    B_contig = B.contiguous().cuda()

    threads = 256
    blocks_per_grid = min(65535, (N * M + threads * 4 - 1) // (threads * 4))

    fused_ext.fused_op(blocks_per_grid, threads, A_contig, B_contig, C, N, M)
    return C
