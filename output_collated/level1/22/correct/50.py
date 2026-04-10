# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_235928/code_10.py
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

# CUDA kernel with float4 vectorization and grid-stride loop for high occupancy
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel_vec4_occ(
    const float* __restrict__ input,
    float*       __restrict__ output,
    const long long numel)
{
    // Grid-stride loop: each thread handles many float4 vectors.
    // Index points to the first element of a float4.
    size_t idx = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const size_t stride = ((size_t)blockDim.x * gridDim.x) * 4;

    for (size_t i = idx; i < numel; i += stride) {
        // Process a full float4 when we have at least 4 items left.
        if (i + 3 < numel) {
            float4 in_vec = reinterpret_cast<const float4*>(input)[i >> 2]; // i/4
            float4 out_vec;
            // Fast math tanh for each component.
            out_vec.x = tanhf(in_vec.x);
            out_vec.y = tanhf(in_vec.y);
            out_vec.z = tanhf(in_vec.z);
            out_vec.w = tanhf(in_vec.w);
            reinterpret_cast<float4*>(output)[i >> 2] = out_vec;
        } else {
            // Tail – fewer than 4 elements remain.
            for (size_t j = i; j < numel; ++j) {
                output[j] = tanhf(input[j]);
            }
            break; // All work done for this thread.
        }
    }
}

// Launcher – chooses a block size that gives high occupancy and limits the grid.
void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const long long numel = input.numel();
    const int threads_per_block = 1024;                     // larger block → more warps
    const int sm_count = 68;                                // RTX 2080 Ti has ~68 SMs
    // Aim for ~8 blocks per SM – a good occupancy compromise.
    int blocks = std::min(static_cast<int>((numel / 4 + threads_per_block - 1) / threads_per_block),
                          sm_count * 8);
    // Clamp to a reasonable maximum (e.g., 4096) to avoid huge grid launch latency.
    const int block_count = (blocks > 4096) ? 4096 : blocks;

    tanh_kernel_vec4_occ<<<block_count, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel);
}
"""

# C++ binding – exposes the kernel to Python
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
          "Vectorized CUDA tanh with grid-stride loop for high occupancy");
}
"""

# Build the inline extension
tanh_ext = load_inline(
    name='tanh_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Model / driver code
def functional_model(x):
    """Functional wrapper required by the evaluation harness."""
    return tanh_ext.custom_tanh(x)

# Configuration (must match the original benchmark)
batch_size = 4096
dim = 393216

def get_init_inputs():
    """No persistent state needed."""
    return []

def get_inputs():
    """Create a fresh input tensor on the GPU."""
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
