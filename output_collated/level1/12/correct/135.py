# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110950/code_7.py
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

__global__ void fused_op_forward_kernel(const float* A, const float* B, float* out, int N, int M) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float a_val = A[row];
        int row_offset = row * M;
        int vec_M = M / 4;
        float4* b_vec = (float4*)&B[row_offset];
        float4* out_vec = (float4*)&out[row_offset];
        
        for (int i = 0; i < vec_M; ++i) {
            float4 b = b_vec[i];
            b.x *= a_val;
            b.y *= a_val;
            b.z *= a_val;
            b.w *= a_val;
            out_vec[i] = b;
        }
    }
}

void fused_op_forward(int blocks, int threads, torch::Tensor A, torch::Tensor B, torch::Tensor out) {
    fused_op_forward_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        out.data_ptr<float>(), 
        A.size(0), 
        B.size(1)
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function in the .cu file
void fused_op_forward(int blocks, int threads, torch::Tensor A, torch::Tensor B, torch::Tensor out);

void launch_fused_op(torch::Tensor A, torch::Tensor B, torch::Tensor out) {
    int N = A.size(0);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    fused_op_forward(blocks, threads, A, B, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_op", &launch_fused_op, "Broadcast multiply kernel");
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
    # A is (N), B is (N, M). 
    # The custom implementation performs the broadcasting multiplication
    out = torch.empty_like(B)
    fused_ext.launch_fused_op(A, B, out)
    return out

# Helpers for testing
N, M = 4096, 4096

def get_inputs():
    A = torch.rand(N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]
