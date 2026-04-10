# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105940/code_21.py
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
    // Each thread process a float4 packet
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < N * M) {
        float4 b_vec = reinterpret_cast<const float4*>(B)[idx / 4];
        
        // Compute row indices for the float4 chunk.
        // Given M is likely a power of 2 or large, division is expensive.
        // If row logic stays the same, we compute explicitly:
        int row0 = idx / M;
        int row1 = (idx + 1) / M;
        int row2 = (idx + 2) / M;
        int row3 = (idx + 3) / M;
        
        // Access A (Broadcast)
        // Since A is small (N), it will hit L1/L2 cache effectively.
        float a0 = A[row0];
        float a1 = A[row1];
        float a2 = A[row2];
        float a3 = A[row3];
        
        float4 out_vec;
        out_vec.x = a0 * b_vec.x;
        out_vec.y = a1 * b_vec.y;
        out_vec.z = a2 * b_vec.z;
        out_vec.w = a3 * b_vec.w;
        
        reinterpret_cast<float4*>(output)[idx / 4] = out_vec;
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    int N = A.size(0);
    int M = B.size(1);
    int total_elements = N * M;
    
    // 128 threads provides high occupancy on 2080Ti
    int threads = 128;
    int blocks = (total_elements + 4 * threads - 1) / (4 * threads);
    
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
void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward, "Vectorized broadcast multiplication");
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
    # Ensure inputs are device-side and contiguous for float4 vector load safety
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    if not A.is_contiguous(): A = A.contiguous()
    if not B.is_contiguous(): B = B.contiguous()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output
