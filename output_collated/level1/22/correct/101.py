# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_004356/code_13.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a Tanh activation.
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

# ----------------------------------------------------------------------
# Optimised CUDA kernel: grid‑stride loop + float4 vectorisation
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void tanh_grid_stride_kernel(const float* __restrict__ x,
                                         float* __restrict__ out,
                                         size_t n) {
    // Global stride = total number of threads * vector width (4)
    const size_t stride = blockDim.x * gridDim.x * 4;
    // Base index for this thread (multiple of 4)
    const size_t base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    // Grid‑stride loop: each thread processes multiple vector chunks
    for (size_t i = base; i < n; i += stride) {
        // Full float4 vector when at least 4 elements remain
        if (i + 3 < n) {
            const float4 vec_x = reinterpret_cast<const float4*>(x)[i / 4];
            float4 vec_out;
            vec_out.x = tanhf(vec_x.x);
            vec_out.y = tanhf(vec_x.y);
            vec_out.z = tanhf(vec_x.z);
            vec_out.w = tanhf(vec_x.w);
            reinterpret_cast<float4*>(out)[i / 4] = vec_out;
        } else {
            // Scalar tail (1‑3 elements left)
            for (size_t j = i; j < n; ++j) {
                out[j] = tanhf(x[j]);
            }
        }
    }
}

// Host routine that decides launch parameters
void fused_tanh_forward(torch::Tensor x, torch::Tensor out) {
    const size_t n = x.numel();               // total number of floats
    const int threads = 256;                  // threads per block
    const int max_blocks = 65535;             // upper limit for grid size

    // Original block count for full coverage (one‑pass)
    int blocks = static_cast<int>((n / 4 + threads - 1) / threads);
    // Cap the block count to avoid excessive launch overhead
    if (blocks > max_blocks) blocks = max_blocks;
    // Ensure we launch at least one block for tiny inputs
    if (blocks == 0) blocks = 1;

    tanh_grid_stride_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n);
}
"""

# ----------------------------------------------------------------------
# C++ binding (pybind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward, "Optimised vectorised tanh forward");
}
"""

# ----------------------------------------------------------------------
# Compile the inline extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_tanh_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional interface
# ----------------------------------------------------------------------
def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Optimised tanh operation.
    Ensures the input is contiguous (required for safe float4 reinterpretation).
    """
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    fused_ext.fused_tanh(x, out)
    return out

def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
