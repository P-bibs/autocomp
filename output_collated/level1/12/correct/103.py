# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_104927/code_26.py
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

// Vectorized grid-stride kernel for maximum memory bandwidth utilization
extern "C"
__global__ void broadcast_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    // 1. Identify the row (one block per row in Y-dimension: gridDim.y = N)
    const int row = blockIdx.y;
    if (row >= N) return;

    // 2. Load the scalar from A once per thread block
    const float a_val = A[row];

    // 3. Grid-stride parameters
    // Thread index within the row/block
    const int tid4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int stride = (gridDim.x * blockDim.x) * 4;

    // 4. Grid-stride loop: processes columns regardless of M divisibility
    for (int col4 = tid4; col4 < (M - (M % 4)); col4 += stride) {
        // Coalesced float4 load
        float4 b_val = reinterpret_cast<const float4*>(&B[row * M + col4])[0];
        
        float4 out_val;
        out_val.x = a_val * b_val.x;
        out_val.y = a_val * b_val.y;
        out_val.z = a_val * b_val.z;
        out_val.w = a_val * b_val.w;
        
        reinterpret_cast<float4*>(&output[row * M + col4])[0] = out_val;
    }

    // Handle remaining elements if M is not divisible by 4
    int remaining = M % 4;
    if (remaining > 0 && threadIdx.x == 0 && blockIdx.x == 0) {
        int start_col = M - remaining;
        for (int i = start_col; i < M; ++i) {
            output[row * M + i] = a_val * B[row * M + i];
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    
    // Launch configuration:
    // One block covers a row, gridDim.y = N, gridDim.x covers columns
    const int threads = 256;
    const int blocks_x = 32; // Sufficient to keep GPU occupied with grid-striding
    dim3 grid(blocks_x, N);
    
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
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output
