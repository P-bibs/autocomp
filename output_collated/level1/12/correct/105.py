# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_104927/code_29.py
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
# CUDA kernel: Optimized Broadcast Multiplication
# ----------------------------------------------------------------------
# Each row is assigned to a block. Threads within the block iterate 
# over the columns with float4 (128-bit) loads. This maximizes 
# memory bandwidth utilization and removes per-element division.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_mul_row_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    const int row = blockIdx.x;
    if (row >= N) return;

    // Load broadcast scalar for the current row
    const float a = A[row];
    
    // Pointer arithmetic for current row
    const float* rowB = B + (size_t)row * M;
    float* rowOut = output + (size_t)row * M;
    
    // Process 4 elements at a time
    const int num_float4 = M / 4;
    const int tid = threadIdx.x;
    
    for (int i = tid; i < num_float4; i += blockDim.x) {
        float4 b_vec = reinterpret_cast<const float4*>(rowB)[i];
        float4 out_vec;
        out_vec.x = a * b_vec.x;
        out_vec.y = a * b_vec.y;
        out_vec.z = a * b_vec.z;
        out_vec.w = a * b_vec.w;
        reinterpret_cast<float4*>(rowOut)[i] = out_vec;
    }
    
    // Tail handling for M not multiple of 4
    if (tid == 0) {
        for (int i = num_float4 * 4; i < M; ++i) {
            rowOut[i] = a * rowB[i];
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    
    // 256 threads per block is generally optimal for memory-bound tasks on 2080Ti
    const int threads = 256;
    const int blocks = N;
    
    broadcast_mul_row_kernel<<<blocks, threads>>>(
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
    m.def("broadcast_mul", &broadcast_mul_forward, "Optimized broadcast multiplication");
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
    Computes output = A.view(-1, 1) * B element-wise.
    Optimized via custom vectorized CUDA kernel.
    """
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    
    # Ensure memory layout requirements for efficient vectorization
    if not B.is_contiguous(): B = B.contiguous()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output

# Helper methods for integration
def get_init_inputs():
    return []

def get_inputs():
    N, M = 4096, 4096
    A = torch.rand(N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]
