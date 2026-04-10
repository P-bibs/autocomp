# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_103853/code_22.py
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

// Using float4 for 128-bit coalesced memory access
__global__ void broadcast_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    // Each thread processes 4 elements to maximize float4 usage
    int m_base = blockIdx.x * (blockDim.x * 4) + threadIdx.x * 4;
    int n = blockIdx.y;

    if (n < N) {
        float a_val = A[n];
        
        // Loop over the row in chunks of 4 floats
        for (int m = m_base; m < M; m += blockDim.x * 4) {
            // Coalesced load
            float4 b_vec = reinterpret_cast<const float4*>(&B[n * M + m])[0];
            
            // Vectorized compute
            float4 out_vec;
            out_vec.x = a_val * b_vec.x;
            out_vec.y = a_val * b_vec.y;
            out_vec.z = a_val * b_vec.z;
            out_vec.w = a_val * b_vec.w;
            
            // Coalesced store
            reinterpret_cast<float4*>(&output[n * M + m])[0] = out_vec;
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    
    // Threads per block. 128 is usually sweet spot for 2080Ti for this compute/load ratio
    const int threads = 128;
    // Each thread handles 4 floats (float4)
    dim3 grid((M / 4 + threads - 1) / threads, N);
    
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
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    # Ensure inputs are contiguous and on CUDA
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    if not A.is_contiguous(): A = A.contiguous()
    if not B.is_contiguous(): B = B.contiguous()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output
