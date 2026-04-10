# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_103853/code_30.py
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

__global__ void broadcast_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    // Shared memory for the broadcast factor A[n]
    __shared__ float a_val;
    
    int n = blockIdx.y;
    // Each block handles one row 'n'. 
    // thread 0 loads the scalar once for the entire block.
    if (threadIdx.x == 0) {
        a_val = A[n];
    }
    __syncthreads();

    // Use float4 for 128-bit wide coalesced access
    int m_idx = blockIdx.x * (blockDim.x * 4) + threadIdx.x * 4;
    
    // Process 4 elements per thread
    if (m_idx <= M - 4) {
        const float4* b_ptr = reinterpret_cast<const float4*>(&B[n * M + m_idx]);
        float4* out_ptr = reinterpret_cast<float4*>(&output[n * M + m_idx]);
        
        float4 b_val = *b_ptr;
        float4 out_val;
        
        float a = a_val;
        out_val.x = a * b_val.x;
        out_val.y = a * b_val.y;
        out_val.z = a * b_val.z;
        out_val.w = a * b_val.w;
        
        *out_ptr = out_val;
    } else {
        // Handle remainder for non-multiple-of-4 widths
        for (int i = m_idx; i < M; ++i) {
            output[n * M + i] = a_val * B[n * M + i];
        }
    }
}

void broadcast_mul(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    
    const int threads = 256;
    // Each thread processes 4 elements via float4
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
void broadcast_mul(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul, "Vectorized broadcast multiply with shared memory");
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

def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are contiguous CUDA tensors
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    if not A.is_contiguous(): A = A.contiguous()
    if not B.is_contiguous(): B = B.contiguous()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output
