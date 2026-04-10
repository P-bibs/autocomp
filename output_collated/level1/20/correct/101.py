# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_220342/code_14.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['negative_slope']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['negative_slope']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a LeakyReLU activation.
    """

    def __init__(self, negative_slope: float=0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope

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
    if 'negative_slope' in flat_state:
        state_kwargs['negative_slope'] = flat_state['negative_slope']
    else:
        state_kwargs['negative_slope'] = getattr(model, 'negative_slope')
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

# --------------------------------------------------------------- #
# Optimised CUDA kernel – uses __ldg (read‑only L2 cache) and
# __stwt (non‑temporal store) to reduce cache pressure.
# --------------------------------------------------------------- #
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Leakily ReLU forward kernel with L2‑cache friendly loads/stores
__global__ void leaky_relu_vectorized_kernel(const float* __restrict__ input,
                                              float* __restrict__ output,
                                              const float negative_slope,
                                              const size_t n)
{
    // Each thread processes 4 elements
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 4 <= n) {
        // ---- read 4 input values through the read‑only L2 cache ----
        const float v0 = __ldg(&input[idx]);
        const float v1 = __ldg(&input[idx + 1]);
        const float v2 = __ldg(&input[idx + 2]);
        const float v3 = __ldg(&input[idx + 3]);

        // ---- apply leaky ReLU (fast‑math multiply) ----
        const float r0 = (v0 > 0.0f) ? v0 : __fmul_rn(v0, negative_slope);
        const float r1 = (v1 > 0.0f) ? v1 : __fmul_rn(v1, negative_slope);
        const float r2 = (v2 > 0.0f) ? v2 : __fmul_rn(v2, negative_slope);
        const float r3 = (v3 > 0.0f) ? v3 : __fmul_rn(v3, negative_slope);

        // ---- non‑temporal write (bypasses L1/L2) ----
        __stwt(&output[idx],     r0);
        __stwt(&output[idx + 1], r1);
        __stwt(&output[idx + 2], r2);
        __stwt(&output[idx + 3], r3);
    }
    else {
        // ---- handle the final few elements (≤3) ----
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const size_t p = idx + i;
            if (p < n) {
                const float v = __ldg(&input[p]);
                const float r = (v > 0.0f) ? v : __fmul_rn(v, negative_slope);
                __stwt(&output[p], r);
            }
        }
    }
}

// Host wrapper that launches the kernel
void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope)
{
    const size_t n = input.numel();
    const int threads = 512;                     // same as original
    const int blocks = (static_cast<size_t>(n) / 4 + threads - 1) / threads;

    leaky_relu_vectorized_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        n
    );
}
"""

# --------------------------------------------------------------- #
# C++ interface / pybind11 binding
# --------------------------------------------------------------- #
cpp_source = r"""
#include <torch/extension.h>

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_forward,
          "Vectorized Leaky ReLU with L2‑cache‑friendly loads/stores");
}
"""

# --------------------------------------------------------------- #
# Build the extension
# --------------------------------------------------------------- #
leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# --------------------------------------------------------------- #
# Functional model – the entry point used for evaluation
# --------------------------------------------------------------- #
def functional_model(x, *, negative_slope):
    """
    Optimised functional_model using the L2‑cache‑friendly CUDA kernel.
    Input is expected to be a CUDA tensor of dtype float32.
    """
    # Ensure we work with contiguous float32 data
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    if not x.is_contiguous():
        x = x.contiguous()

    output = torch.empty_like(x)          # same device & dtype as x
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    return output


# --------------------------------------------------------------- #
# Boiler‑plate required by the evaluation harness
# --------------------------------------------------------------- #
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []   # no additional initialisation needed

def get_inputs():
    # Generate a random input tensor on the GPU
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
