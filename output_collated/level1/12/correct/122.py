# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105940/code_18.py
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

// Optimized kernel: Uses register caching for A[n] and ensures perfect coalescing
__global__ void broadcast_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    // Each row n is assigned to a unique Y-block. 
    // Threads in the same row share the same value A[n].
    int n = blockIdx.y;
    if (n >= N) return;

    // By loading A[n] once into a register, we minimize global memory access.
    // Modern GPUs will automatically broadcast this load if coded correctly.
    float a_val = A[n];
    
    int m_base = blockIdx.x * (blockDim.x * 4);
    
    for (int i = threadIdx.x * 4; i < M; i += blockDim.x * 4) {
        int idx = m_base + i;
        if (idx + 3 < M) {
            float4 b_val = reinterpret_cast<const float4*>(&B[n * M + idx])[0];
            float4 out_val;
            out_val.x = a_val * b_val.x;
            out_val.y = a_val * b_val.y;
            out_val.z = a_val * b_val.z;
            out_val.w = a_val * b_val.w;
            reinterpret_cast<float4*>(&output[n * M + idx])[0] = out_val;
        } else {
            // Cleanup for non-multiple of 4 widths
            for (int k = 0; k < 4 && (idx + k) < M; ++k) {
                output[n * M + idx + k] = a_val * B[n * M + idx + k];
            }
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    
    // 128 threads is often optimal for keeping occupancy high while allowing enough register pressure
    const int threads = 128;
    // Each thread handles 4 floats (float4)
    dim3 grid((M + (threads * 4) - 1) / (threads * 4), N);
    
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
    m.def("broadcast_mul", &broadcast_mul_forward, "Vectorized broadcast multiplication");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3'],
    with_cuda=True
)

def functional_model(A, B):
    # Ensure inputs are contiguous and on cuda
    A = A.contiguous().cuda()
    B = B.contiguous().cuda()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output
