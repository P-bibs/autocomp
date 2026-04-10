# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110950/code_28.py
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

__global__ void fused_op_forward_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M
) {
    // Each thread processes 4 elements (1 vector) at a time.
    // We use a grid-stride loop to handle matrices of arbitrary size.
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    int total_vectors = (N * M) / 4;
    
    for (int i = idx; i < total_vectors; i += blockDim.x * gridDim.x) {
        int row = (i * 4) / M;
        int col = (i * 4) % M;
        
        float a_val = A[row];
        float4 b_vec = reinterpret_cast<const float4*>(&B[i * 4])[0];
        float4 c_vec;
        
        c_vec.x = a_val * b_vec.x;
        c_vec.y = a_val * b_vec.y;
        c_vec.z = a_val * b_vec.z;
        c_vec.w = a_val * b_vec.w;
        
        reinterpret_cast<float4*>(&C[i * 4])[0] = c_vec;
    }

    // Handle tail if M is not a multiple of 4
    for (int i = idx + total_vectors * 4; i < N * M; i += blockDim.x * gridDim.x) {
        int row = i / M;
        C[i] = A[row] * B[i];
    }
}

void fused_op_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int N, int M) {
    const int threads = 256;
    // Limit grid size to max-threads-per-SM * number of SMs (roughly)
    // 2080Ti has 68 SMs, 65535 blocks is plenty for coverage
    int total_vectors = (N * M) / 4;
    int blocks = (total_vectors + threads - 1) / threads;
    blocks = std::min(blocks, 65535);

    fused_op_forward_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int N, int M);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Optimized coalesced fused multiply");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    N, M = B.shape
    # Ensure inputs are contiguous to satisfy __restrict__ and float4 alignment
    A = A.contiguous().cuda()
    B = B.contiguous().cuda()
    C = torch.empty(N, M, dtype=torch.float32, device='cuda')
    
    fused_ext.fused_op(A, B, C, N, M)
    return C
