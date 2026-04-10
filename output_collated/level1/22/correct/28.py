# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_233710/code_12.py
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
# CUDA kernel – grid‑stride elementwise tanh with float4 vectorization
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            const int64_t numel) {
    // Grid‑stride loop: each thread processes multiple elements
    int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x * 4;

    // Process full float4 vectors while we have at least 4 items left
    while (idx + 3 < numel) {
        float4 in = reinterpret_cast<const float4*>(input)[idx >> 2];
        float4 out;
        out.x = tanhf(in.x);
        out.y = tanhf(in.y);
        out.z = tanhf(in.z);
        out.w = tanhf(in.w);
        reinterpret_cast<float4*>(output)[idx >> 2] = out;
        idx += stride;
    }

    // Handle any leftover elements (up to three)
#pragma unroll
    for (int i = 0; i < 4 && idx + i < numel; ++i) {
        output[idx + i] = tanhf(input[idx + i]);
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int64_t numel = input.numel();
    const int threads_per_block = 256;
    const int blocks = 8192;                     // modest block count, sufficient parallelism

    tanh_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding – exposes the custom tanh operation to Python
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output);

torch::Tensor custom_tanh(const torch::Tensor& input) {
    auto output = torch::empty_like(input);
    launch_tanh_kernel(input, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_tanh", &custom_tanh, "Grid‑stride tanh CUDA kernel");
}
"""

# -------------------------------------------------------------------------
# Compile the extension
# -------------------------------------------------------------------------
tanh_ext = load_inline(
    name='tanh_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Model interface
# -------------------------------------------------------------------------
def functional_model(x):
    return tanh_ext.custom_tanh(x)

# -------------------------------------------------------------------------
# Inputs for evaluation harness
# -------------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Input must be on GPU (CUDA) as required by the evaluation
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
