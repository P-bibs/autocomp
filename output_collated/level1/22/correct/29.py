# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_233710/code_13.py
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
# CUDA source – kernel + host wrapper
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void tanh_kernel(const float* __restrict__ x,
                            float*       __restrict__ out,
                            size_t n)
{
    // Total number of threads in the grid – used for the grid‑stride loop
    const size_t stride = blockDim.x * gridDim.x;
    // Starting position for this thread
    size_t start = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid‑stride loop: each thread processes many, non‑overlapping elements
    for (size_t idx = start; idx < n; idx += stride) {
        // Vectorised path: handle 4 elements at once when possible
        if (idx + 3 < n) {
            // Read through the read‑only data cache ( __ldg )
            const float x0 = __ldg(x + idx);
            const float x1 = __ldg(x + idx + 1);
            const float x2 = __ldg(x + idx + 2);
            const float x3 = __ldg(x + idx + 3);

            // Fast hardware‑intrinsic tanh
            float y0 = tanhf(x0);
            float y1 = tanhf(x1);
            float y2 = tanhf(x2);
            float y3 = tanhf(x3);

            // Store the results
            out[idx]     = y0;
            out[idx + 1] = y1;
            out[idx + 2] = y2;
            out[idx + 3] = y3;
        } else {
            // Clean‑up loop for the last 1‑3 elements
            for (size_t j = idx; j < n; ++j) {
                out[j] = tanhf(__ldg(x + j));
            }
        }
    }
}

// Host‑side wrapper that chooses a reasonable grid size
void fused_tanh_forward(torch::Tensor x, torch::Tensor out)
{
    const size_t n = x.numel();
    const int threads = 256;                     // one block contains 256 threads

    // Query the device to obtain the number of SMs
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    // Aim for ~16 blocks per SM, but cap the launch at 4096 blocks
    int blocks = prop.multiProcessorCount * 16;
    if (blocks > 4096) blocks = 4096;
    if (blocks <= 0) blocks = 1;                 // safety fallback

    // Launch the kernel
    tanh_kernel<<<blocks, threads>>>(x.data_ptr<float>(),
                                     out.data_ptr<float>(),
                                     n);
}
"""

# -------------------------------------------------------------------------
# C++ binding (PyBind11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

// Declaration of the CUDA‑implemented function
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward,
          "Optimized tanh forward using read‑only cache");
}
"""

# -------------------------------------------------------------------------
# Build the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_tanh',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True,
)

# -------------------------------------------------------------------------
# User‑visible functions
# -------------------------------------------------------------------------
def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized element‑wise tanh.
    The input is guaranteed to be contiguous before the kernel is launched.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)          # allocate output on the same device
    fused_ext.fused_tanh(x, out)       # launch the CUDA kernel
    return out


def get_init_inputs():
    """No persistent state is required."""
    return []


def get_inputs():
    """Generate a large batch that matches the benchmark size."""
    batch_size = 4096
    dim = 393216
    # Use rand to get a realistic input (values in [0,1))
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
