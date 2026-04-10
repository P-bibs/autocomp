# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_211408/code_28.py
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
# Constants 
# ----------------------------------------------------------------------
batch_size = 4096
dim = 393216

# ----------------------------------------------------------------------
# Optimised CUDA kernel (Grid-stride + Vectorized float4 + __ldg)
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Vectorized Leaky ReLU Kernel using Grid-Stride Loops
// This pattern allows for high occupancy and optimal memory coalescing
__global__ void leaky_relu_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        float negative_slope,
        size_t num_elements) 
{
    // Total number of float4 vectors
    const size_t vec_size = num_elements / 4;
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop: each thread processes multiple float4 segments
    for (; id < vec_size; id += blockDim.x * gridDim.x) {
        const float4 in_vec = __ldg(reinterpret_cast<const float4*>(input) + id);
        float4 out_vec;
        
        // Fast branch-less FMA implementation
        out_vec.x = fmaf(negative_slope, fminf(in_vec.x, 0.0f), fmaxf(in_vec.x, 0.0f));
        out_vec.y = fmaf(negative_slope, fminf(in_vec.y, 0.0f), fmaxf(in_vec.y, 0.0f));
        out_vec.z = fmaf(negative_slope, fminf(in_vec.z, 0.0f), fmaxf(in_vec.z, 0.0f));
        out_vec.w = fmaf(negative_slope, fminf(in_vec.w, 0.0f), fmaxf(in_vec.w, 0.0f));
        
        reinterpret_cast<float4*>(output)[id] = out_vec;
    }

    // Handle tail elements (if any)
    size_t tail_idx = vec_size * 4 + (blockIdx.x * blockDim.x + threadIdx.x);
    for (; tail_idx < num_elements; tail_idx += blockDim.x * gridDim.x) {
        const float val = input[tail_idx];
        output[tail_idx] = fmaf(negative_slope, fminf(val, 0.0f), fmaxf(val, 0.0f));
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int threads = 256;
    // Heuristic: Use enough blocks to keep SMs busy without excessive grid overhead
    const int blocks = 1024; 

    leaky_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        n
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_forward, "Grid-stride vectorized leaky relu");
}
"""

leaky_relu_ext = load_inline(
    name='leaky_relu_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimised functional model using grid-stride kernel execution.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    return output
