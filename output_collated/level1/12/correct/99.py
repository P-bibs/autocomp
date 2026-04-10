# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_104927/code_22.py
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

# The kernel uses float4 vectorization and loop unrolling to maximize memory bandwidth
# and register usage on an RTX 2080Ti. By processing 16 floats (4x float4) per thread,
# we increase ILP and allow the GPU to better hide memory latency.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    // Each thread processes 16 floats (4 float4 vectors)
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int m_idx = tid_x * 16;
    int n = blockIdx.y;

    if (n < N && m_idx + 15 < M) {
        float a_val = A[n];
        const float* b_row = &B[n * M];
        float* out_row = &output[n * M];
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float4 b_val = reinterpret_cast<const float4*>(&b_row[m_idx + i * 4])[0];
            
            float4 out_val;
            out_val.x = a_val * b_val.x;
            out_val.y = a_val * b_val.y;
            out_val.z = a_val * b_val.z;
            out_val.w = a_val * b_val.w;
            
            reinterpret_cast<float4*>(&out_row[m_idx + i * 4])[0] = out_val;
        }
    } else if (n < N && m_idx < M) {
        // Handle remainder for non-multiple-of-16 dimensions
        float a_val = A[n];
        for (int i = m_idx; i < M; ++i) {
            output[n * M + i] = a_val * B[n * M + i];
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    
    // 128 threads per block as per optimization plan
    const int threads = 128;
    // Each thread processes 16 floats
    dim3 grid((M / 16 + threads - 1) / threads, N);
    
    broadcast_mul_kernel<<<grid, threads>>>(
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
    m.def("broadcast_mul", &broadcast_mul_forward, "Vectorized and unrolled broadcast multiplication");
}
"""

# Compile the extension inline
fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Optimized broadcast multiplication: A (N,) * B (N, M) -> (N, M)
    """
    # Ensure inputs are contiguous on device
    A = A.contiguous().cuda()
    B = B.contiguous().cuda()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output
