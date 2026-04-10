# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112116/code_10.py
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

// Grid-stride loop version for better performance and flexibility
__global__ void broadcast_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process multiple float4 elements per thread using grid-stride loop
    for (int n = blockIdx.y; n < N; n += gridDim.y) {
        const float a_val = A[n];
        
        // Each thread processes elements at indices: tid*4, (tid + stride)*4, (tid + 2*stride)*4, ...
        for (int m_base = tid * 4; m_base < M; m_base += stride * 4) {
            // Handle boundary conditions to avoid out-of-bounds access
            if (m_base + 3 < M) {
                // Full float4 can be processed
                float4 b_val = reinterpret_cast<const float4*>(&B[n * M + m_base])[0];
                
                float4 out_val;
                out_val.x = a_val * b_val.x;
                out_val.y = a_val * b_val.y;
                out_val.z = a_val * b_val.z;
                out_val.w = a_val * b_val.w;
                
                reinterpret_cast<float4*>(&output[n * M + m_base])[0] = out_val;
            } else {
                // Process remaining elements individually to handle non-multiple-of-4 sizes
                for (int k = 0; k < 4 && m_base + k < M; ++k) {
                    output[n * M + m_base + k] = a_val * B[n * M + m_base + k];
                }
            }
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    
    // Use fixed block size for consistent performance
    const int threads_per_block = 256;
    const int elements_per_thread = 4;
    
    // Grid size calculation that works for any M value
    const int blocks_x = (M + elements_per_thread * threads_per_block - 1) / (elements_per_thread * threads_per_block);
    const dim3 grid(blocks_x, min(N, 65535)); // Limit grid.y to maximum allowed value
    
    broadcast_mul_kernel<<<grid, threads_per_block>>>(
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
    m.def("broadcast_mul", &broadcast_mul_forward, "Vectorized broadcast multiplication with grid-stride loop");
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
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output

# Test parameters
N, M = 4096, 4096

def get_inputs():
    return [torch.rand(N).cuda(), torch.rand(N, M).cuda()]

