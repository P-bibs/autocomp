# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_102701/code_24.py
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

# ------------------------------------------------------------------
#   CUDA kernels
#   Optimized: Removed divergent branches by decoupling the 
#   vectorized bulk from the tail.
# ------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_mul_vec4_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ out,
    const int    N,
    const int    M,
    const int    total_vec4)
{
    const int vec4_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec4_id >= total_vec4) return;

    const int idx = vec4_id * 4;
    const float4 b_vec = reinterpret_cast<const float4*>(B)[vec4_id];

    // Compute row indices for each element in the float4
    // Using simple division as M is constant.
    const int row0 = (idx) / M;
    const int row1 = (idx + 1) / M;
    const int row2 = (idx + 2) / M;
    const int row3 = (idx + 3) / M;

    const float4 a_vec = make_float4(A[row0], A[row1], A[row2], A[row3]);

    const float4 out_vec = make_float4(
        a_vec.x * b_vec.x,
        a_vec.y * b_vec.y,
        a_vec.z * b_vec.z,
        a_vec.w * b_vec.w
    );

    reinterpret_cast<float4*>(out)[vec4_id] = out_vec;
}

__global__ void broadcast_mul_tail_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ out,
    const int    M,
    const int    tail_start,
    const int    tail_len)
{
    const int i = threadIdx.x;
    if (i < tail_len) {
        int idx = tail_start + i;
        out[idx] = A[idx / M] * B[idx];
    }
}

void launch_broadcast_mul(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& out) {
    const int N = A.size(0);
    const int M = B.size(1);
    const int total_elements = N * M;
    const int total_vec4 = total_elements / 4;
    
    const int threads = 256;
    if (total_vec4 > 0) {
        const int blocks = (total_vec4 + threads - 1) / threads;
        broadcast_mul_vec4_kernel<<<blocks, threads>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), N, M, total_vec4
        );
    }
    
    const int tail_start = total_vec4 * 4;
    const int tail_len = total_elements - tail_start;
    if (tail_len > 0) {
        broadcast_mul_tail_kernel<<<1, tail_len>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), M, tail_start, tail_len
        );
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_broadcast_mul(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &launch_broadcast_mul, "Optimized branch-free broadcast mul");
}
"""

fused_ext = load_inline(
    name='broadcast_mul_opt',
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
