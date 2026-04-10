# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_185153/code_16.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a ReLU activation.
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
# CUDA source (Optimized Fused Kernel)
# -------------------------------------------------------------------------
# Optimization:
# 1. Single kernel launch reduces dispatch overhead.
# 2. Each thread processes a float4 (4 floats) for peak memory throughput.
# 3. Only the final active thread handles boundary scalars to ensure correctness 
#    without warp divergence across the bulk of the data.
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_fused_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  size_t n) {
    size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t element_idx = vec_idx * 4;

    // Fast path: Process full float4 vectors
    if (element_idx + 3 < n) {
        float4 in_vec = __ldg(reinterpret_cast<const float4*>(input) + vec_idx);
        float4 out_vec;
        out_vec.x = fmaxf(in_vec.x, 0.0f);
        out_vec.y = fmaxf(in_vec.y, 0.0f);
        out_vec.z = fmaxf(in_vec.z, 0.0f);
        out_vec.w = fmaxf(in_vec.w, 0.0f);
        reinterpret_cast<float4*>(output)[vec_idx] = out_vec;
    } 
    // Boundary path: Handle remaining 1-3 elements
    else if (element_idx < n) {
        for (int i = 0; i < 4; ++i) {
            size_t idx = element_idx + i;
            if (idx < n) {
                output[idx] = fmaxf(input[idx], 0.0f);
            }
        }
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    size_t n = input.numel();
    if (n == 0) return;

    // Grid size: ceil(n / 4) threads total
    // Max blocks is sufficient to cover (n/4)
    size_t total_vecs = (n + 3) / 4;
    int threads = 256;
    int blocks = (total_vecs + threads - 1) / threads;

    relu_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        n
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused ReLU kernel");
}
"""

# -------------------------------------------------------------------------
# Build the inline extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Applies ReLU on `x` using a single-pass vectorized CUDA kernel.
    Input `x` is assumed to be a contiguous float32 tensor on GPU.
    """
    output = torch.empty_like(x)
    fused_ext.fused_op(x, output)
    return output

# Benchmark-related helpers
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]
