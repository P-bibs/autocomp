# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112116/code_26.py
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
# CUDA kernel – optimized with a grid-stride loop
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Vectorized grid-stride kernel for maximum memory bandwidth and latency hiding
__global__ void broadcast_mul_kernel_grid_stride(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    const int N,
    const int M
) {
    // Total elements in the M dimension (each is a float)
    // We stride based on the number of threads * 4 (since each thread handles a float4)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride over rows: allows handling N > gridDim.y
    for (int n = blockIdx.y; n < N; n += gridDim.y) {
        float a_val = A[n];
        
        // Grid-stride over columns: each thread handles multiple float4 chunks
        for (int i = tid; i < M / 4; i += stride) {
            int offset = n * M + i * 4;
            
            // Load, Multiply, Store using float4 for coalesced 128-bit memory access
            float4 b_val = reinterpret_cast<const float4*>(&B[offset])[0];
            
            float4 out_val;
            out_val.x = a_val * b_val.x;
            out_val.y = a_val * b_val.y;
            out_val.z = a_val * b_val.z;
            out_val.w = a_val * b_val.w;
            
            reinterpret_cast<float4*>(&output[offset])[0] = out_val;
        }

        // Handle remaining elements if M is not a multiple of 4
        if (M % 4 != 0 && tid == 0) {
            for (int m = (M / 4) * 4; m < M; ++m) {
                output[n * M + m] = a_val * B[n * M + m];
            }
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    
    // Launch configuration: 256 threads per block, 
    // Grid x covers columns, Grid y covers rows
    const int threads = 256;
    const int blocks_x = std::min((M / 4 + threads - 1) / threads, 1024);
    const int blocks_y = std::min(N, 128);
    
    dim3 grid(blocks_x, blocks_y);
    
    broadcast_mul_kernel_grid_stride<<<grid, threads>>>(
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
    m.def("broadcast_mul", &broadcast_mul_forward, "Vectorized grid-stride broadcast multiplication");
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
    Optimized broadcast multiplication using a grid-stride CUDA kernel.
    """
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output
