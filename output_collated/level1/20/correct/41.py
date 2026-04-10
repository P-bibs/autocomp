# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_204858/code_15.py
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

# ----------------------------------------------------------------------
# CUDA kernel – now with grid‑stride loop, larger block size,
# __restrict__ pointers, and optional __ldg for the read‑only input.
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Vectorised Leaky ReLU kernel – grid‑stride version
__global__ void leaky_relu_vectorized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float negative_slope,
    const size_t n)
{
    // Each thread works on a vector of 4 elements
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    // Grid‑stride step: all threads together cover the whole tensor
    const size_t stride = blockDim.x * gridDim.x * 4;

    for (size_t i = idx; i < n; i += stride) {
        // Process a full float4 vector when possible
        if (i + 4 <= n) {
            // Coalesced 128‑bit load
            float4 vals = *reinterpret_cast<const float4*>(&input[i]);

            // Fast leaky ReLU with fused multiply‑add (fast‑math)
            vals.x = (vals.x > 0.0f) ? vals.x : __fmul_rn(vals.x, negative_slope);
            vals.y = (vals.y > 0.0f) ? vals.y : __fmul_rn(vals.y, negative_slope);
            vals.z = (vals.z > 0.0f) ? vals.z : __fmul_rn(vals.z, negative_slope);
            vals.w = (vals.w > 0.0f) ? vals.w : __fmul_rn(vals.w, negative_slope);

            // Coalesced 128‑bit store
            *reinterpret_cast<float4*>(&output[i]) = vals;
        } else {
            // Tail – fewer than 4 elements remaining
            for (size_t j = i; j < n; ++j) {
                float val = input[j];
                output[j] = (val > 0.0f) ? val : __fmul_rn(val, negative_slope);
            }
        }
    }
}

// Host‑side launch wrapper
void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();

    // Use 1024 threads per block (max occupancy on RTX 2080 Ti)
    const int threads = 1024;
    // A modest grid size – 8192 blocks gives >8 M active threads.
    // This is enough to keep all SMs busy while keeping the launch overhead low.
    const int blocks = 8192;

    leaky_relu_vectorized_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        n
    );
}
"""

# ----------------------------------------------------------------------
# C++ interface (pybind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_forward, "Vectorized Leaky ReLU forward kernel");
}
"""

# ----------------------------------------------------------------------
# Compile the extension with aggressive optimisation flags
# ----------------------------------------------------------------------
leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional model – now uses the improved kernel
# ----------------------------------------------------------------------
def functional_model(x, *, negative_slope):
    """
    Optimised functional_model using a grid‑stride, high‑occupancy CUDA kernel.
    Each thread processes many float4 vectors to amortise launch overhead.
    """
    # Ensure input is float32 and contiguous (required by the kernel)
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    if not x.is_contiguous():
        x = x.contiguous()

    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    return output

# ----------------------------------------------------------------------
# Helpers required by the benchmark harness
# ----------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Float32 input on GPU – matches kernel expectations
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
