# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_204858/code_30.py
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
# CUDA Kernel implementation
# The kernel uses float4 vectorization for memory bandwidth, 
# __ldg for cache efficiency, and branch-less FMA math for throughput.
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float leaky_relu_op(float val, float slope) {
    // Branch-less implementation of Leaky ReLU:
    // fmaxf(val, 0) + slope * fminf(val, 0)
    // fmaf performs (a * b) + c efficiently.
    return fmaf(slope, fminf(val, 0.0f), fmaxf(val, 0.0f));
}

__global__ void leaky_relu_vectorized_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        float negative_slope,
        size_t n) 
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idx = id * 4;

    if (idx + 3 < n) {
        // Load through texture cache (read-only)
        float4 in_vec = __ldg(reinterpret_cast<const float4*>(input) + id);
        float4 out_vec;

        out_vec.x = leaky_relu_op(in_vec.x, negative_slope);
        out_vec.y = leaky_relu_op(in_vec.y, negative_slope);
        out_vec.z = leaky_relu_op(in_vec.z, negative_slope);
        out_vec.w = leaky_relu_op(in_vec.w, negative_slope);

        reinterpret_cast<float4*>(output)[id] = out_vec;
    } else {
        // Cleanup loop for non-aligned elements
        for (int i = 0; i < 4 && idx + i < n; ++i) {
            output[idx + i] = leaky_relu_op(input[idx + i], negative_slope);
        }
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int threads = 256;
    const int blocks = (n / 4 + threads - 1) / threads;

    leaky_relu_vectorized_kernel<<<blocks, threads>>>(
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
    m.def("leaky_relu", &leaky_relu_forward, "Vectorized Leaky ReLU (Branchless/ldg)");
}
"""

# Compile the extension
leaky_relu_ext = load_inline(
    name='leaky_relu_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized functional_model using vectorized CUDA kernel.
    Ensures input is contiguous and f32, then invokes the custom kernel.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    
    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    return output

# --- Helper functions required by test harness ---
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
