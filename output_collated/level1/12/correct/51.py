# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_102701/code_28.py
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

# -------------------------------------------------------------------------
# Optimized CUDA kernel
# The primary optimization involves loading the broadcast value A into a register
# once per thread to avoid redundant global memory accesses. We use vectorized
# float4 loads/stores to maximize throughput.
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void broadcast_mul_vectorized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    // Each thread processes 4 elements (1 float4)
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // Total number of elements
    int total_elements = N * M;
    
    if (idx + 3 < total_elements) {
        // Fast Path: Check if all 4 elements are within the same row (A index)
        // If (idx / M) == ((idx + 3) / M), they share the same A value.
        int row0 = idx / M;
        int row3 = (idx + 3) / M;
        
        if (row0 == row3) {
            float val_a = A[row0];
            float4 b_vec = reinterpret_cast<const float4*>(B)[idx / 4];
            
            float4 out_vec;
            out_vec.x = val_a * b_vec.x;
            out_vec.y = val_a * b_vec.y;
            out_vec.z = val_a * b_vec.z;
            out_vec.w = val_a * b_vec.w;
            
            reinterpret_cast<float4*>(output)[idx / 4] = out_vec;
        } else {
            // Edge Case: The 4 elements span across two rows.
            // Still process element-wise but strictly within registers.
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int cur = idx + i;
                if (cur < total_elements) {
                    output[cur] = A[cur / M] * B[cur];
                }
            }
        }
    } else {
        // Cleanup loop for remaining elements if total_elements is not divisible by 4
        for (int i = idx; i < total_elements; ++i) {
            output[i] = A[i / M] * B[i];
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    const int total_elements = N * M;
    
    // 256 threads is standard for optimal occupancy on 2080Ti
    const int threads_per_block = 256;
    const int blocks = (total_elements / 4 + threads_per_block - 1) / threads_per_block;
    
    broadcast_mul_vectorized_kernel<<<blocks, threads_per_block>>>(
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
    m.def("broadcast_mul", &broadcast_mul_forward, "Optimized vectorized broadcast multiplication");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='broadcast_mul_ext_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Computes A (N) * B (N, M) -> output (N, M) broadcasted over columns.
    Optimized to minimize global memory bandwidth by caching A in registers.
    """
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output

M = 4096
N = 4096

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(N, device='cuda', dtype=torch.float32)
    B = torch.rand(N, M, device='cuda', dtype=torch.float32)
    return [A, B]
