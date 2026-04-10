# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_185153/code_13.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a ReLU activation.
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
# Custom CUDA kernel: vectorized ReLU with a grid‑stride loop
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_vec_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                const size_t num_elements) {
    // Grid‑stride: each thread processes multiple 4‑element chunks
    const size_t stride = (size_t)blockDim.x * gridDim.x * 4ULL;
    for (size_t idx = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4ULL;
         idx < num_elements;
         idx += stride) {
        // Full float4 load/store when we have at least 4 elements left
        if (idx + 3 < num_elements) {
            float4 in_vec = reinterpret_cast<const float4*>(input)[idx >> 2]; // idx/4
            float4 out_vec;
            out_vec.x = fmaxf(0.0f, in_vec.x);
            out_vec.y = fmaxf(0.0f, in_vec.y);
            out_vec.z = fmaxf(0.0f, in_vec.z);
            out_vec.w = fmaxf(0.0f, in_vec.w);
            reinterpret_cast<float4*>(output)[idx >> 2] = out_vec;
        } else {
            // Handle the remaining 1‑3 elements
            for (size_t i = idx; i < num_elements; ++i) {
                output[i] = fmaxf(0.0f, input[i]);
            }
        }
    }
}

void launch_relu_kernel(const float* input, float* output, const size_t num_elements) {
    const int threads_per_block = 256;
    // Use a moderate grid size; the grid‑stride loop will cover the whole tensor
    const int num_blocks = 1024;
    relu_vec_kernel<<<num_blocks, threads_per_block>>>(input, output, num_elements);
}
"""

# ----------------------------------------------------------------------
# C++ binding (PyBind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void launch_relu_kernel(const float* input, float* output, const size_t num_elements);

void relu_forward(torch::Tensor input, torch::Tensor output) {
    launch_relu_kernel(input.data_ptr<float>(), output.data_ptr<float>(), input.numel());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_forward", &relu_forward, "Vectorized ReLU forward pass");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension
# ----------------------------------------------------------------------
relu_ext = load_inline(
    name='relu_op_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional wrapper
# ----------------------------------------------------------------------
def functional_model(x):
    # Ensure a contiguous layout for vectorized float4 access
    if not x.is_contiguous():
        x = x.contiguous()
    output = torch.empty_like(x)
    relu_ext.relu_forward(x, output)
    return output

# ----------------------------------------------------------------------
# Benchmark helpers (unchanged)
# ----------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
