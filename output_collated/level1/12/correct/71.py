# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_103853/code_21.py
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

# The optimization strategy is to use vector-load/store (float4) combined with 
# the 2D grid strategy to ensure maximum throughput and memory coalescence. 
# We remove all integer division inside the loop by iterating per-row and 
# handling M-sized chunks with block-level parallelism.

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
    // Each block handles one row (threadIdx.x processes M elements)
    int n = blockIdx.x;
    if (n >= N) return;

    float a_val = A[n];
    
    // Process M elements using vector4 loads to maximize memory bandwidth
    int m = threadIdx.x * 4;
    for (; m + 3 < M; m += blockDim.x * 4) {
        int idx = n * M + m;
        
        float4 b_vec = reinterpret_cast<const float4*>(B + idx)[0];
        float4 out_vec;
        
        out_vec.x = a_val * b_vec.x;
        out_vec.y = a_val * b_vec.y;
        out_vec.z = a_val * b_vec.z;
        out_vec.w = a_val * b_vec.w;
        
        reinterpret_cast<float4*>(output + idx)[0] = out_vec;
    }

    // Handle trailing elements if M is not a multiple of 4
    for (; m < M; m += blockDim.x) {
        output[n * M + m] = a_val * B[n * M + m];
    }
}

void broadcast_mul(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    
    // We use M/4 threads to utilize vectorization, capped at 256
    int threads_x = (M + 3) / 4;
    threads_x = (threads_x > 256) ? 256 : threads_x;
    
    dim3 blocks(N);
    dim3 threads(threads_x);
    
    broadcast_mul_kernel<<<blocks, threads>>>(
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
    m.def("broadcast_mul", &broadcast_mul, "Optimized vectorized broadcast multiplication");
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
    # Ensure inputs are on the correct device
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    
    # Create the output tensor
    output = torch.empty_like(B)
    
    # Launch kernel
    fused_ext.broadcast_mul(A, B, output)
    
    return output

# Inputs as required
N = 4096
M = 4096

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]
