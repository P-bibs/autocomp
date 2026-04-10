# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_220342/code_24.py
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
# Parameters
# ----------------------------------------------------------------------
THREADS_PER_BLOCK = 128
ITEMS_PER_THREAD = 8  # 8 * 4 = 32 floats per thread for better ILP
ELEMENTS_PER_LOAD = 4 # float4

# ----------------------------------------------------------------------
# CUDA Kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float leaky_relu_op(float x, float slope) {
    return fmaf(slope, fminf(x, 0.0f), fmaxf(x, 0.0f));
}

__global__ void leaky_relu_optimized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float slope,
    size_t n) 
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    
    // Each thread processes a chunk of work
    size_t idx = tid * 8; 
    
    for (size_t i = idx; i < n; i += stride * 8) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            size_t curr = i + j;
            if (curr < n) {
                output[curr] = leaky_relu_op(input[curr], slope);
            }
        }
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int threads = 256;
    // Utilize grid-stride loop for balance and scalability
    const int blocks = std::min((int)((n + (threads * 8) - 1) / (threads * 8)), 65535);

    leaky_relu_optimized_kernel<<<blocks, threads>>>(
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
    m.def("leaky_relu", &leaky_relu_forward, "Optimized Leaky ReLU");
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

# ----------------------------------------------------------------------
# Functional Model
# ----------------------------------------------------------------------
_cached_output = None

def functional_model(x, *, negative_slope):
    """
    Optimized Leaky ReLU using a grid-stride loop kernel.
    This approach ensures full GPU occupancy and balances work across
    all SMs regardless of input size, while minimizing launch overhead.
    """
    global _cached_output

    if not x.is_contiguous():
        x = x.contiguous()

    # Reuse output buffer to avoid dealloc/alloc overhead
    if _cached_output is None or _cached_output.shape != x.shape or _cached_output.device != x.device:
        _cached_output = torch.empty_like(x)

    leaky_relu_ext.leaky_relu(x, _cached_output, float(negative_slope))
    
    return _cached_output
