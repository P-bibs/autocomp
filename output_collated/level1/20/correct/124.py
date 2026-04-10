# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_222822/code_13.py
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
# CUDA kernel – 8 floats per thread (two float4 loads)
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_vectorized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float negative_slope,
    size_t n)
{
    // Each thread processes 8 consecutive floats (two float4 vectors)
    size_t idx = (size_t)(blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const float zero = 0.0f;

    if (idx + 7 < n) {
        // Vectorised loads: two float4
        float4 in0 = reinterpret_cast<const float4*>(input)[idx / 4];
        float4 in1 = reinterpret_cast<const float4*>(input)[idx / 4 + 1];
        float4 out0, out1;

        // Leaky ReLU for the 8 elements
        out0.x = (in0.x > zero) ? in0.x : in0.x * negative_slope;
        out0.y = (in0.y > zero) ? in0.y : in0.y * negative_slope;
        out0.z = (in0.z > zero) ? in0.z : in0.z * negative_slope;
        out0.w = (in0.w > zero) ? in0.w : in0.w * negative_slope;
        out1.x = (in1.x > zero) ? in1.x : in1.x * negative_slope;
        out1.y = (in1.y > zero) ? in1.y : in1.y * negative_slope;
        out1.z = (in1.z > zero) ? in1.z : in1.z * negative_slope;
        out1.w = (in1.w > zero) ? in1.w : in1.w * negative_slope;

        // Vectorised stores
        reinterpret_cast<float4*>(output)[idx / 4] = out0;
        reinterpret_cast<float4*>(output)[idx / 4 + 1] = out1;
    } else {
        // Tail handling (up to 8 elements)
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            size_t cur = idx + i;
            if (cur < n) {
                float val = input[cur];
                output[cur] = (val > zero) ? val : val * negative_slope;
            }
        }
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int block_size = 1024;
    // Grid now covers 8 elements per thread
    const int grid_size = (n + (block_size * 8) - 1) / (block_size * 8);

    leaky_relu_vectorized_kernel<<<grid_size, block_size>>>(
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
    m.def("leaky_relu", &leaky_relu_forward, "Vectorized Leaky ReLU forward (8 elements per thread)");
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
    """
    Optimized functional_model using the compiled CUDA kernel.
    """
    # Ensure contiguous float32 data
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
