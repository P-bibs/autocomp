# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_233710/code_15.py
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

# -------------------------------------------------------------------------
# CUDA kernel with grid‑stride loops (optimisation #7)
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <math.h>

__global__ void fused_tanh_kernel(const float* __restrict__ x,
                                   float*       __restrict__ out,
                                   const size_t n)
{
    // Grid‑stride loop: each thread processes multiple float4 chunks.
    const size_t stride = blockDim.x * gridDim.x * 4;          // elements per loop
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4; // start index

    for (size_t i = idx; i < n; i += stride) {
        const size_t remaining = n - i;
        if (remaining >= 4) {
            // Vectorised load/store via float4
            float4 vec_x = reinterpret_cast<const float4*>(x)[i >> 2];
            float4 vec_out;
            vec_out.x = tanhf(vec_x.x);
            vec_out.y = tanhf(vec_x.y);
            vec_out.z = tanhf(vec_x.z);
            vec_out.w = tanhf(vec_x.w);
            reinterpret_cast<float4*>(out)[i >> 2] = vec_out;
        } else {
            // Handle the tail (1‑3 elements) scalar‑wise
            for (size_t j = 0; j < remaining; ++j) {
                out[i + j] = tanhf(x[i + j]);
            }
        }
    }
}

// Host function that chooses a modest grid (≈8 blocks per SM) and launches the kernel
void fused_tanh_forward(torch::Tensor x, torch::Tensor out)
{
    const size_t n          = x.numel();
    const int    threads    = 256;                     // threads per block
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, x.device().index());

    // Target ~8 blocks per multiprocessor for good occupancy
    int blocks = prop.multiProcessorCount * 8;
    if (blocks < 1) blocks = 1;

    fused_tanh_kernel<<<blocks, threads>>>(x.data_ptr<float>(),
                                           out.data_ptr<float>(),
                                           n);
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward,
          "Vectorized tanh forward using grid‑stride loops");
}
"""

# -------------------------------------------------------------------------
# Compile the inline extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_tanh',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Public functional model (the only entry point used for evaluation)
# -------------------------------------------------------------------------
def functional_model(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    fused_ext.fused_tanh(x, out)
    return out


# -------------------------------------------------------------------------
# Test configuration (kept exactly as in the original script)
# -------------------------------------------------------------------------
batch_size = 4096
dim = 393216


def get_init_inputs():
    return []


def get_inputs():
    # Create a contiguous float32 tensor on GPU
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
