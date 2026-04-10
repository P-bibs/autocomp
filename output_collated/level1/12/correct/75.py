# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_103853/code_24.py
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

# CUDA Kernel with Grid-Stride Loop
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void fused_op_forward_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M
) {
    const int total = N * M;
    const int stride = gridDim.x * blockDim.x;
    // Each thread starts processing 4 elements
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    for (; i < total; i += stride * 4) {
        int row = i / M;
        float a_val = A[row];
        int remaining_in_row = (row + 1) * M - i;

        if (remaining_in_row >= 4) {
            // Fully coalesced float4 load/store
            float4 b_vec = reinterpret_cast<const float4*>(&B[i])[0];
            float4 c_vec;
            c_vec.x = a_val * b_vec.x;
            c_vec.y = a_val * b_vec.y;
            c_vec.z = a_val * b_vec.z;
            c_vec.w = a_val * b_vec.w;
            reinterpret_cast<float4*>(&C[i])[0] = c_vec;
        } else {
            // Tail handling
            #pragma unroll
            for (int j = 0; j < remaining_in_row; ++j) {
                C[i + j] = a_val * B[i + j];
            }
        }
    }
}

void fused_op_forward(int blocks, int threads, const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int N, int M) {
    fused_op_forward_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );
}
"""

# C++ Wrapper
cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(int blocks, int threads, const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int N, int M);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Vectorized grid-stride fused multiply");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Optimized fused unsqueeze-multiply using grid-stride loops.
    A is [N], B is [N, M] -> Output [N, M]
    """
    N, M = B.shape
    C = torch.empty(N, M, dtype=torch.float32, device='cuda')
    
    A_contig = A.contiguous().cuda()
    B_contig = B.contiguous().cuda()
    
    threads = 256
    # Calculate blocks per grid-stride loop requirements
    total_elements = N * M
    blocks = (total_elements + (threads * 4 - 1)) // (threads * 4)
    # Ensure at least 1 block to avoid empty kernel launch
    blocks = max(1, min(blocks, 65535))
    
    fused_ext.fused_op(blocks, threads, A_contig, B_contig, C, N, M)
    
    return C
