# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112116/code_22.py
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

// Optimized kernel using Grid-Stride Loops and float4 vectorization
__global__ void broadcast_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    const long total_elements,
    const int M
) {
    // Process 4 floats at a time
    long idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    long stride = (long)blockDim.x * gridDim.x * 4;

    for (long i = idx; i < total_elements; i += stride) {
        // Ensure we don't overflow the float4 load if total_elements is not a multiple of 4
        if (i + 3 < total_elements) {
            float4 b_val = reinterpret_cast<const float4*>(&B[i])[0];
            float4 out_val;
            
            // Broadcast A[n] for each element in the float4
            // Since we are iterating over a flattened array, we need the row index n
            // Optimization: if i and i+3 are in the same row, compute n once
            int n_start = i / M;
            int n_end = (i + 3) / M;
            
            if (n_start == n_end) {
                float a = A[n_start];
                out_val.x = a * b_val.x;
                out_val.y = a * b_val.y;
                out_val.z = a * b_val.z;
                out_val.w = a * b_val.w;
            } else {
                // Handle rare case where float4 crosses row boundary
                out_val.x = A[i / M] * B[i];
                out_val.y = A[(i + 1) / M] * B[i + 1];
                out_val.z = A[(i + 2) / M] * B[i + 2];
                out_val.w = A[(i + 3) / M] * B[i + 3];
            }
            reinterpret_cast<float4*>(&output[i])[0] = out_val;
        } else {
            // Cleanup loop for remaining elements
            for (long j = i; j < total_elements; ++j) {
                output[j] = A[j / M] * B[j];
            }
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const long N = A.size(0);
    const long M = B.size(1);
    const long total_elements = N * M;
    
    // Launch configuration
    const int threads = 256;
    const int blocks = 1024; // Sufficient to saturate SMs on 2080Ti
    
    broadcast_mul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        (int)M
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward, "Vectorized broadcast multiplication with grid-stride loops");
}
"""

fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    # Ensure inputs are contiguous on GPU
    A = A.contiguous()
    B = B.contiguous()
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output
