# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_202725/code_14.py
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
# CUDA source – the Leaky‑ReLU kernel + wrapper
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Grid‑strided kernel that reads each element once and writes the result once.
__global__ void leaky_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float negative_slope,
    int64_t N)
{
    // Each thread processes one element; the loop strides across the whole tensor.
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < N; i += stride) {
        float x = __ldg(&input[i]);               // cached read
        float y = (x > 0.0f) ? x : negative_slope * x; // branch‑less leaky ReLU
        output[i] = y;                            // write back
    }
}

// C++ wrapper callable from Python
void leaky_relu_cuda(torch::Tensor input, torch::Tensor output, float negative_slope)
{
    int64_t N = input.numel();
    const int BLOCK_DIM = 256;
    // Use the maximum number of blocks (65535) to keep the grid small while
    // still covering the whole tensor with the strided loop.
    int grid = (N + BLOCK_DIM - 1) / BLOCK_DIM;
    if (grid > 65535) grid = 65535;

    leaky_relu_kernel<<<grid, BLOCK_DIM>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        N);
    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
# C++ source – PyBind11 binding
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void leaky_relu_cuda(torch::Tensor input, torch::Tensor output, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_cuda, "Leaky ReLU CUDA kernel");
}
"""

# ----------------------------------------------------------------------
# Compile the extension
# ----------------------------------------------------------------------
leaky_ext = load_inline(
    name='leaky_relu',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Model parameters (same as the original script)
# ----------------------------------------------------------------------
batch_size = 4096
dim = 393216

# ----------------------------------------------------------------------
# functional_model – the function that will be imported
# ----------------------------------------------------------------------
def functional_model(x, *, negative_slope):
    """
    Applies Leaky‑ReLU to the whole tensor using a custom CUDA kernel.
    The kernel reads each element exactly once and writes the result once,
    which minimises global‑memory traffic and eliminates the original CPU
    overhead.
    """
    # Ensure the input is on GPU
    if not x.is_cuda:
        x = x.cuda()

    # Allocate output on the same device
    out = torch.empty_like(x)

    # Call the hand‑written CUDA kernel
    leaky_ext.leaky_relu(x, out, negative_slope)

    return out

# ----------------------------------------------------------------------
# Helper functions that may be used by the test harness
# ----------------------------------------------------------------------
def get_init_inputs():
    return []  # No special initialisation required


def get_inputs():
    # Create the input on the GPU directly to avoid an extra copy
    x = torch.rand(batch_size, dim, dtype=torch.float32, device='cuda')
    return [x]
