# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_213956/code_12.py
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
# CUDA kernel – grid‑stride loop, block size 256, limited grid
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_grid_stride_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float negative_slope,
    size_t n)
{
    // Grid‑stride loop: each thread processes multiple elements
    size_t idx = (size_t)(blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const size_t stride = (size_t)blockDim.x * gridDim.x * 4;

    for (size_t i = idx; i < n; i += stride) {
        // Vectorised load when we have a full float4
        if (i + 3 < n) {
            float4 in_vec = reinterpret_cast<const float4*>(input)[i / 4];
            float4 out_vec;
            out_vec.x = (in_vec.x > 0.0f) ? in_vec.x : in_vec.x * negative_slope;
            out_vec.y = (in_vec.y > 0.0f) ? in_vec.y : in_vec.y * negative_slope;
            out_vec.z = (in_vec.z > 0.0f) ? in_vec.z : in_vec.z * negative_slope;
            out_vec.w = (in_vec.w > 0.0f) ? in_vec.w : in_vec.w * negative_slope;
            reinterpret_cast<float4*>(output)[i / 4] = out_vec;
        } else {
            // Tail: handle at most 3 remaining elements
            for (size_t j = 0; j < 4 && (i + j) < n; ++j) {
                float val = input[i + j];
                output[i + j] = (val > 0.0f) ? val : val * negative_slope;
            }
        }
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();

    // 256 threads per block gives better occupancy than 1024
    const int block_size = 256;
    // Fixed modest grid (1024 blocks) – the grid‑stride loop covers the whole array
    const int grid_size = 1024;

    leaky_relu_grid_stride_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        n
    );
}
"""

# ----------------------------------------------------------------------
# C++ binding
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_forward, "Grid‑stride Leaky ReLU forward");
}
"""

# Compile the extension
leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """Optimised functional_model using a grid‑stride CUDA kernel."""
    if not x.is_contiguous():
        x = x.contiguous()
    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    return output

# ----------------------------------------------------------------------
# Evaluation helpers (unchanged)
# ----------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
