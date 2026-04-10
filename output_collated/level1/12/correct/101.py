# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_104927/code_21.py
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

// Vectorized broadcast multiplication with minimized integer division
__global__ void broadcast_mul_vectorized_optimized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    const int N,
    const int M
) {
    // Each thread processes 4 floats (float4)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 4;
    
    // We process M elements per row. 
    // To avoid division: row = idx / M
    // Since we process contiguous memory, we look for which row boundaries we cross.
    if (idx + 3 < N * M) {
        float4 b_vec = reinterpret_cast<const float4*>(B)[tid];
        
        // Calculate rows for the 4 elements
        int n0 = idx / M;
        int n1 = (idx + 1) / M;
        int n2 = (idx + 2) / M;
        int n3 = (idx + 3) / M;
        
        float4 out_vec;
        out_vec.x = A[n0] * b_vec.x;
        out_vec.y = A[n1] * b_vec.y;
        out_vec.z = A[n2] * b_vec.z;
        out_vec.w = A[n3] * b_vec.w;
        
        reinterpret_cast<float4*>(output)[tid] = out_vec;
    } else {
        // Scalar cleanup for edge cases
        for (int i = idx; i < N * M; ++i) {
            output[i] = A[i / M] * B[i];
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    const int total_elements = N * M;
    
    // Threads per block
    const int threads = 256;
    // Each thread handles 4 elements = 1024 elements per block
    const int blocks = (total_elements + 4 * threads - 1) / (4 * threads);
    
    broadcast_mul_vectorized_optimized_kernel<<<blocks, threads>>>(
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
    name='broadcast_mul_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    # Ensure inputs are contiguous and on GPU
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    A = A.contiguous()
    B = B.contiguous()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output
