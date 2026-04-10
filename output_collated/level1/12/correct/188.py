# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112116/code_31.py
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

# -------------------------------------------------------------------------
# CUDA source – Optimized row-wise scaling kernel
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Vectorized load/store for better bandwidth utilization (128-bit)
__global__ void row_scale_kernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float*       __restrict__ C,
                                 int N, int M) {
    const int i = blockIdx.x;
    const float a = A[i];
    
    // Process columns in blocks of 4 using float4 for vectorized memory access
    // This improves coalescing efficiency by increasing the number of bytes 
    // transferred per memory transaction.
    int col = threadIdx.x * 4;
    
    // Loop through elements in chunks of 4 * blockDim.x
    for (int j = col; j < M; j += blockDim.x * 4) {
        if (j + 3 < M) {
            float4 b_val = reinterpret_cast<const float4*>(&B[i * M + j])[0];
            float4 out;
            out.x = a * b_val.x;
            out.y = a * b_val.y;
            out.z = a * b_val.z;
            out.w = a * b_val.w;
            reinterpret_cast<float4*>(&C[i * M + j])[0] = out;
        } else {
            // Handle remaining columns if M is not a multiple of 4
            for (int k = j; k < M; ++k) {
                C[i * M + k] = a * B[i * M + k];
            }
        }
    }
}

void row_scale(at::Tensor A, at::Tensor B, at::Tensor C) {
    const int N = B.size(0);
    const int M = B.size(1);
    
    // 128 threads per block allows enough throughput, 
    // each thread handles 4 floats at a time
    const int threads = 128;
    const int blocks = N;

    row_scale_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void row_scale(at::Tensor A, at::Tensor B, at::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("row_scale", &row_scale, "Optimized row-wise scaling");
}
"""

# -------------------------------------------------------------------------
# Build the extension
# -------------------------------------------------------------------------
row_scale_ext = load_inline(
    name="row_scale_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)

def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized broadcast multiply (C = A[:, None] * B).
    """
    N, M = B.shape
    C = torch.empty((N, M), dtype=torch.float32, device=A.device)
    
    # Launch kernel via binding
    row_scale_ext.row_scale(A, B, C)
    
    return C

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
N = 4096
M = 4096

def get_init_inputs():
    return []

def get_inputs():
    # Ensure tensors are contiguous and on correct device
    A = torch.rand(N, device="cuda", dtype=torch.float32)
    B = torch.rand(N, M, device="cuda", dtype=torch.float32)
    return [A, B]
