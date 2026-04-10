# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_104927/code_10.py
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

// Vectorized kernel using grid-stride loop for maximum memory bandwidth utilization
extern "C"
__global__ void broadcast_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ out,
    int N, int M)
{
    // 1. Identify the row we are working on.
    const int row = blockIdx.y;
    if (row >= N) return;

    // 2. Load the scalar from A once per thread block (cached in a register)
    const float a_val = A[row];

    // 3. Compute the global thread index that addresses float4 elements.
    const int tid4   = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int stride = (gridDim.x * blockDim.x) * 4;

    // 4. Grid-stride loop over the columns.
    for (int col4 = tid4; col4 < M; col4 += stride) {
        // Load 4 consecutive B values (coalesced)
        float4 b_vec = reinterpret_cast<const float4*>(B + row * M + col4)[0];

        // Element-wise multiplication with the broadcasted scalar
        float4 out_vec;
        out_vec.x = a_val * b_vec.x;
        out_vec.y = a_val * b_vec.y;
        out_vec.z = a_val * b_vec.z;
        out_vec.w = a_val * b_vec.w;

        // Store the result (coalesced)
        reinterpret_cast<float4*>(out + row * M + col4)[0] = out_vec;
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);

    // Thread block configuration
    const int threads_per_block = 256;
    
    // Grid dimensions
    const int blocks_x = (M + threads_per_block * 4 - 1) / (threads_per_block * 4);
    const int blocks_y = N;
    
    dim3 grid(blocks_x, blocks_y);

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

# Test parameters matching the request
N, M = 4096, 4096

def get_inputs():
    return [torch.rand(N).cuda(), torch.rand(N, M).cuda()]
