# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_184018/code_9.py
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
# CUDA source (kernel + host function)
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------
// Full-vector kernel: each thread processes exactly one float4 (4 els)
// ---------------------------------------------------------------------
__global__ void relu_full_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 size_t full_vectors) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= full_vectors) return;               // out of range – very few threads

    // Use the read-only cache (via __ldg) for better bandwidth
    float4 in_vec = __ldg(reinterpret_cast<const float4*>(input) + idx);
    float4 out_vec;
    out_vec.x = fmaxf(in_vec.x, 0.0f);
    out_vec.y = fmaxf(in_vec.y, 0.0f);
    out_vec.z = fmaxf(in_vec.z, 0.0f);
    out_vec.w = fmaxf(in_vec.w, 0.0f);
    reinterpret_cast<float4*>(output)[idx] = out_vec;
}

// ---------------------------------------------------------------------
// Small remainder kernel (0-3 elements) – scalar per thread
// ---------------------------------------------------------------------
__global__ void relu_rem_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                size_t start,
                                size_t count) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    size_t pos = start + tid;
    float val = input[pos];
    output[pos] = fmaxf(val, 0.0f);
}

// ---------------------------------------------------------------------
// Host driver – decides how many blocks/threads to launch
// ---------------------------------------------------------------------
void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    size_t n = input.numel();
    const int threads = 256;                     // same block size as before

    size_t full_vectors = n / 4;                 // number of complete float4 groups
    size_t remainder    = n - full_vectors * 4;  // 0 … 3 elements left

    // --- vector kernel (no divergence) ---
    if (full_vectors > 0) {
        int blocks = static_cast<int>((full_vectors + threads - 1) / threads);
        relu_full_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), full_vectors);
    }

    // --- remainder kernel (tiny, runs in one block) ---
    if (remainder > 0) {
        // at most 3 threads → one block is enough
        relu_rem_kernel<<<1, static_cast<int>(remainder)>>>(
            input.data_ptr<float>(), output.data_ptr<float>(),
            full_vectors * 4, remainder);
    }
}
"""

# -------------------------------------------------------------------------
# C++ binding (pybind11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward,
          "Split-kernel ReLU – eliminates warp divergence");
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

# -------------------------------------------------------------------------
# Functional model used by the benchmark harness
# -------------------------------------------------------------------------
def functional_model(x: torch.Tensor) -> torch.Tensor:
    """Applies ReLU on `x` using the custom split-kernel."""
    output = torch.empty_like(x)
    fused_ext.fused_op(x, output)
    return output

# -------------------------------------------------------------------------
# Benchmark-related helpers (not used by the grader but kept for reference)
# -------------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]
