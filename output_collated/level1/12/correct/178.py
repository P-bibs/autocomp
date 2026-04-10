# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112116/code_21.py
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

# The CUDA kernel is optimized for memory coalescing. 
# By assigning one block per matrix row, we ensure that 'A[row]' is fetched once 
# into a register and reused across the entire row of 'B'. 
# This eliminates the overhead of calculating indices (no division/modulus) 
# and ensures that threads in a warp access contiguous memory segments, 
# maximizing bus bandwidth utilization on the RTX 2080Ti.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_mul_coalesced_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    int row = blockIdx.x;
    if (row >= N) return;

    // Load the broadcast factor into a register for the entire block
    float a_val = A[row];
    const float* row_B = B + (row * M);
    float* row_out = output + (row * M);

    // Each thread processes 4 elements (1 float4) to maximize throughput
    // and satisfy coalesced memory access requirements.
    int tid = threadIdx.x;
    int stride = blockDim.x * 4;
    
    for (int i = tid * 4; i < M; i += stride) {
        // Load contiguous float4 from global memory
        float4 b_vec = reinterpret_cast<const float4*>(&row_B[i])[0];
        
        // Element-wise arithmetic
        float4 out_vec;
        out_vec.x = a_val * b_vec.x;
        out_vec.y = a_val * b_vec.y;
        out_vec.z = a_val * b_vec.z;
        out_vec.w = a_val * b_vec.w;
        
        // Store contiguous float4 to global memory
        reinterpret_cast<float4*>(&row_out[i])[0] = out_vec;
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    
    // We use 256 threads per block. Each block handles 1 row of M elements.
    dim3 grid(N);
    dim3 block(256);
    
    broadcast_mul_coalesced_kernel<<<grid, block>>>(
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
    m.def("broadcast_mul", &broadcast_mul_forward, "Vectorized coalesced broadcast multiplication");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='broadcast_mul_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Optimized functional model using a custom CUDA kernel to perform 
    row-wise broadcast multiplication with coalesced memory access.
    """
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    
    # Ensure inputs are contiguous
    if not A.is_contiguous(): A = A.contiguous()
    if not B.is_contiguous(): B = B.contiguous()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output
