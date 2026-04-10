# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_001958/code_12.py
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
# Optimized CUDA kernel: no shared memory, no syncthreads, direct store
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int numel) {
    // Each thread handles up to 4 consecutive elements (float4 vectorization)
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;

    if (idx + 3 < numel) {
        // Fully vectorized load-compute-store
        float4 in_vec = reinterpret_cast<const float4*>(input)[idx / 4];
        float4 out_vec;
        out_vec.x = tanhf(in_vec.x);
        out_vec.y = tanhf(in_vec.y);
        out_vec.z = tanhf(in_vec.z);
        out_vec.w = tanhf(in_vec.w);
        reinterpret_cast<float4*>(output)[idx / 4] = out_vec;
    } else {
        // Boundary case: handle remaining elements one-by-one
        for (int i = 0; i < 4; ++i) {
            int offset = idx + i;
            if (offset < numel) {
                output[offset] = tanhf(input[offset]);
            }
        }
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int numel = input.numel();
    const int threads_per_block = 256;                 // multiple of 32, good occupancy
    const int elements_per_block = threads_per_block * 4; // 1024 elements per block
    const int blocks = (numel + elements_per_block - 1) / elements_per_block;

    tanh_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel
    );
}
"""

# ----------------------------------------------------------------------
# C++ binding (PYBIND11) to expose the kernel to Python
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
    m.def("custom_tanh", &custom_tanh, "Optimized CUDA tanh without shared memory or syncs");
}
"""

# ----------------------------------------------------------------------
# Compile the extension with aggressive optimization flags
# ----------------------------------------------------------------------
tanh_ext = load_inline(
    name='tanh_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Model wrapper required by the evaluation harness
# ----------------------------------------------------------------------
def functional_model(x):
    """Applies the custom tanh kernel to the input tensor."""
    return tanh_ext.custom_tanh(x)

# ----------------------------------------------------------------------
# Benchmark configuration (matching the original problem size)
# ----------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    """No persistent state needed."""
    return []

def get_inputs():
    """Generate a random GPU tensor of the expected shape."""
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
