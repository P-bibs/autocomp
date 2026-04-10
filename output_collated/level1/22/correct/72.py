# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_001958/code_14.py
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
# CUDA kernel – vectorized tanh with loop unrolling (4 float4 per thread)
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel_vec4_unroll(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int numel) {
    const int unroll = 4;                     // number of float4 vectors per thread
    const int per_thread = 4 * unroll;        // 16 elements per thread
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * per_thread;

    #pragma unroll
    for (int j = 0; j < unroll; ++j) {
        const int offset = idx + j * 4;       // element offset for this inner iteration
        if (offset + 3 < numel) {             // fully‑occupied float4 chunk
            float4 in_vec = reinterpret_cast<const float4*>(input)[offset / 4];
            float4 out_vec;
            out_vec.x = tanhf(in_vec.x);
            out_vec.y = tanhf(in_vec.y);
            out_vec.z = tanhf(in_vec.z);
            out_vec.w = tanhf(in_vec.w);
            reinterpret_cast<float4*>(output)[offset / 4] = out_vec;
        } else {                              // tail – handle remaining elements
            for (int k = offset; k < numel; ++k) {
                output[k] = tanhf(input[k]);
            }
            break;                            // all remaining work done
        }
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int numel = input.numel();
    const int threads_per_block = 256;
    const int unroll = 4;
    const int per_thread = 4 * unroll;                 // 16 elements per thread
    const int total_threads = (numel + per_thread - 1) / per_thread;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    tanh_kernel_vec4_unroll<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel
    );
}
"""

# -------------------------------------------------------------------------
# C++ bindings (PYBIND11)
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
    m.def("custom_tanh", &custom_tanh,
          "Vectorized CUDA tanh implementation with loop unrolling");
}
"""

# -------------------------------------------------------------------------
# Build the extension
# -------------------------------------------------------------------------
tanh_ext = load_inline(
    name='tanh_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional model that will be evaluated
# -------------------------------------------------------------------------
def functional_model(x):
    """Apply the custom tanh kernel to input tensor x."""
    return tanh_ext.custom_tanh(x)

# -------------------------------------------------------------------------
# Interface required by the evaluation harness
# -------------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    """No persistent state required."""
    return []

def get_inputs():
    """Return a random GPU tensor of the correct shape."""
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
