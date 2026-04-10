# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_211408/code_8.py
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
# Constants (kept for compatibility with the reference test harness)
# ----------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Create a random input tensor of the required shape and dtype
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]

# ----------------------------------------------------------------------
# Optimised CUDA kernel (unrolled + increased work per thread)
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define ELEMENTS_PER_THREAD 16
#define UNROLL_FACTOR (ELEMENTS_PER_THREAD / 4)

__global__ void leaky_relu_vectorized_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        float negative_slope,
        size_t n) 
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idx = tid * ELEMENTS_PER_THREAD;
    
    if (idx + ELEMENTS_PER_THREAD <= n) {
        // Process multiple float4 elements per thread with loop unrolling
        const float4* input4 = reinterpret_cast<const float4*>(input);
        float4* output4 = reinterpret_cast<float4*>(output);
        
        float4 in_vec[UNROLL_FACTOR];
        float4 out_vec[UNROLL_FACTOR];
        
        // Software pipelining: load all data first
        #pragma unroll
        for (int i = 0; i < UNROLL_FACTOR; i++) {
            in_vec[i] = __ldg(input4 + tid * UNROLL_FACTOR + i);
        }
        
        // Process all data with branchless operations
        #pragma unroll
        for (int i = 0; i < UNROLL_FACTOR; i++) {
            out_vec[i].x = fmaf(negative_slope, fminf(in_vec[i].x, 0.0f), fmaxf(in_vec[i].x, 0.0f));
            out_vec[i].y = fmaf(negative_slope, fminf(in_vec[i].y, 0.0f), fmaxf(in_vec[i].y, 0.0f));
            out_vec[i].z = fmaf(negative_slope, fminf(in_vec[i].z, 0.0f), fmaxf(in_vec[i].z, 0.0f));
            out_vec[i].w = fmaf(negative_slope, fminf(in_vec[i].w, 0.0f), fmaxf(in_vec[i].w, 0.0f));
        }
        
        // Write all results
        #pragma unroll
        for (int i = 0; i < UNROLL_FACTOR; i++) {
            output4[tid * UNROLL_FACTOR + i] = out_vec[i];
        }
    } else {
        // Handle boundary conditions
        for (int i = 0; i < ELEMENTS_PER_THREAD && idx + i < n; ++i) {
            const float val = input[idx + i];
            const float out = fmaf(negative_slope, fminf(val, 0.0f), fmaxf(val, 0.0f));
            output[idx + i] = out;
        }
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int threads = 256;
    // Adjust blocks calculation for increased work per thread
    const int blocks = (n / ELEMENTS_PER_THREAD + threads - 1) / threads;

    leaky_relu_vectorized_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        n
    );
}
"""

# ----------------------------------------------------------------------
# C++ binding (PYBIND11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_forward,
          "Vectorized Leaky ReLU forward (optimised with loop unrolling and increased work per thread)");
}
"""

# ----------------------------------------------------------------------
# Build the inline extension
# ----------------------------------------------------------------------
leaky_relu_ext = load_inline(
    name='leaky_relu_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional model – entry point used by the evaluator
# ----------------------------------------------------------------------
def functional_model(x, *, negative_slope):
    """
    Optimised functional model using a custom CUDA kernel.
    The kernel uses:
      * __ldg to load read-only data through the texture cache,
      * loop unrolling and increased work per thread for better ILP,
      * branch-less Leaky ReLU (fmaxf / fminf + fmaf) for lower latency.
    """
    # Ensure the input is contiguous and of the correct type
    if not x.is_contiguous():
        x = x.contiguous()
    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    # Allocate output (re-used across calls if desired)
    output = torch.empty_like(x)

    # Launch the optimised kernel
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    return output
