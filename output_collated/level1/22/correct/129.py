# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_010905/code_14.py
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
# CUDA kernel – grid‑stride, float4 vectorised, one block per batch element
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel_vec4(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int dim) {
    // One block per batch element
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;
    const int stride = blockDim.x * 4;                // stride in floats
    const int row_start = row * dim;
    const int row_end   = row_start + dim;

    // Grid‑stride loop: each thread processes many float4 vectors
    for (int idx = row_start + tid * 4; idx < row_end; idx += stride) {
        // Whole float4 fits into the row?
        if (idx + 4 <= row_end) {
            const int offset = idx >> 2;               // idx/4
            // Read through L1/texture cache
            float4 in_vec = __ldg(reinterpret_cast<const float4*>(input) + offset);
            float4 out_vec;
            out_vec.x = tanhf(in_vec.x);
            out_vec.y = tanhf(in_vec.y);
            out_vec.z = tanhf(in_vec.z);
            out_vec.w = tanhf(in_vec.w);
            reinterpret_cast<float4*>(output)[offset] = out_vec;
        } else {
            // Handle the last 1‑3 elements scalar‑wise
            for (int i = idx; i < row_end; ++i) {
                output[i] = tanhf(input[i]);
            }
        }
    }
}

// Launch wrapper – decides grid size based on batch dimension
void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int batch = input.size(0);          // number of rows
    const int dim   = input.size(1);          // elements per row
    const int threads_per_block = 256;        // multiple of 32
    const int blocks = batch;                 // one block per batch element

    tanh_kernel_vec4<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        dim
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding – exposes the custom tanh to Python
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
    m.def("custom_tanh", &custom_tanh, "Vectorized CUDA tanh implementation");
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
# Functional interface – the model that will be evaluated
# -------------------------------------------------------------------------
def functional_model(x):
    """Apply the custom vectorised tanh."""
    return tanh_ext.custom_tanh(x)

# -------------------------------------------------------------------------
# Benchmark configuration (must match the evaluation harness)
# -------------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    """No persistent state is required."""
    return []

def get_inputs():
    """Return a random GPU tensor of the proper shape."""
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
