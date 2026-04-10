# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_012911/code_12.py
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
# CUDA kernel – direct global-memory vectorized tanh (no shared memory)
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Each thread handles 4 consecutive floats (float4) for maximum memory
// coalescing and instruction-level parallelism.
__global__ void tanh_kernel_direct(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int numel)
{
    // Compute the global index of the first element this thread will process.
    // blockDim.x = 256, each thread processes 4 elements.
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    // Vectorized path – all 4 elements are within bounds.
    if (idx + 3 < numel) {
        float4 in = reinterpret_cast<const float4*>(input)[idx / 4];
        float4 out;
        out.x = tanhf(in.x);
        out.y = tanhf(in.y);
        out.z = tanhf(in.z);
        out.w = tanhf(in.w);
        reinterpret_cast<float4*>(output)[idx / 4] = out;
    } else {
        // Scalar tail – handle up to 3 remaining elements.
        for (int i = 0; i < 4; ++i) {
            if (idx + i < numel) {
                output[idx + i] = tanhf(input[idx + i]);
            }
        }
    }
}

// Launcher that configures the grid and hands the tensors to the kernel.
void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int numel = input.numel();
    const int threads_per_block = 256;          // 256 threads per block
    const int elements_per_thread = 4;           // float4 vectorization
    const int tile_size = threads_per_block * elements_per_thread; // 1024
    const int blocks = (numel + tile_size - 1) / tile_size;

    tanh_kernel_direct<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel);
}
"""

# ----------------------------------------------------------------------
# C++ binding – exposes the kernel to Python via PyTorch's CPP extension
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output);

torch::Tensor custom_tanh(const torch::Tensor& input) {
    auto output = torch::empty_like(input);
    launch_tanh_kernel(input, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_tanh", &custom_tanh,
          "Direct-access vectorized CUDA tanh kernel");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension
# ----------------------------------------------------------------------
tanh_ext = load_inline(
    name='tanh_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Model entry point
# ----------------------------------------------------------------------
def functional_model(x):
    """Applies element-wise tanh using the custom CUDA kernel."""
    return tanh_ext.custom_tanh(x)

# ----------------------------------------------------------------------
# Benchmark / test configuration (matches the original problem size)
# ----------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    """No persistent state is required."""
    return []

def get_inputs():
    """Create a random input tensor on the GPU."""
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
