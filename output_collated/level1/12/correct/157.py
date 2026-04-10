# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110950/code_29.py
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
# CUDA kernel – tiled, vectorised broadcast multiplication
# ------------------------------------------------------------
# Optimization details:
# 1. Row parity: Used blockIdx.y to identify rows, eliminating integer division.
# 2. Vectorized Loads/Stores: Each thread processes 4 floats (float4) to ensure coalesced memory access.
# 3. Memory Coalescing: float4 ensures 128-bit loads, maximizing bus utilization.
# 4. Reduced Global Traffic: A[row] is loaded once per block into a register.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_mul_tile_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    const int row = blockIdx.y;
    if (row >= N) return;

    // Load broadcast value once per block
    const float a = A[row];

    // Each thread processes 4 elements
    const int tid = threadIdx.x;
    const int col_base = blockIdx.x * (blockDim.x * 4);
    const int col = col_base + tid * 4;

    // Use pointers to the specific row
    const float* __restrict__ row_B = B + row * M;
    float* __restrict__ row_out = output + row * M;

    // Vectorized path
    if (col + 3 < M) {
        float4 b_vec = reinterpret_cast<const float4*>(row_B + col)[0];
        float4 out_vec;
        out_vec.x = a * b_vec.x;
        out_vec.y = a * b_vec.y;
        out_vec.z = a * b_vec.z;
        out_vec.w = a * b_vec.w;
        reinterpret_cast<float4*>(row_out + col)[0] = out_vec;
    } else {
        // Handle tail elements
        for (int i = 0; i < 4; ++i) {
            if (col + i < M) {
                row_out[col + i] = a * row_B[col + i];
            }
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);

    // Threads per block set to 128 (processes 512 floats per block)
    const int threads_per_block = 128;
    const int tile_width = threads_per_block * 4;
    const int blocks_x = (M + tile_width - 1) / tile_width;
    const int blocks_y = N;

    dim3 grid(blocks_x, blocks_y);
    broadcast_mul_tile_kernel<<<grid, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        M
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward, "Vectorized broadcast multiplication");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Optimized broadcast multiplication: A (N) * B (N, M).
    """
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output

N = 4096
M = 4096

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]
