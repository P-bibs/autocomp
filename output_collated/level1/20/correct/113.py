# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_220342/code_27.py
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
# Optimized CUDA kernel using Grid-Stride Loops and float4 vectorization.
# Each thread processes multiple chunks of data to maximize memory 
# throughput and hide latency on the RTX 2080Ti.
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void leaky_relu_grid_stride_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        float negative_slope,
        size_t n_vec) 
{
    // Each thread processes elements in a grid-stride loop
    // Using float4 for 128-bit coalesced memory access
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Convert to float4 indices
    for (size_t i = tid; i < n_vec; i += blockDim.x * gridDim.x) {
        float4 in_vec = reinterpret_cast<const float4*>(input)[i];
        
        float4 out_vec;
        // Perform Leaky ReLU on each component
        // Using fma for efficiency and fast_math for hardware-level fmin/max
        out_vec.x = (in_vec.x > 0.0f) ? in_vec.x : (negative_slope * in_vec.x);
        out_vec.y = (in_vec.y > 0.0f) ? in_vec.y : (negative_slope * in_vec.y);
        out_vec.z = (in_vec.z > 0.0f) ? in_vec.z : (negative_slope * in_vec.z);
        out_vec.w = (in_vec.w > 0.0f) ? in_vec.w : (negative_slope * in_vec.w);
        
        reinterpret_cast<float4*>(output)[i] = out_vec;
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const size_t n_vec = n / 4;
    
    // Heuristic for block/grid configuration
    const int threads = 512;
    // Cap blocks to allow flexibility for occupancy
    const int max_blocks = 65535; 
    const int blocks = std::min(max_blocks, (int)((n_vec + threads - 1) / threads));

    leaky_relu_grid_stride_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        n_vec
    );
    
    // Handle remaining elements if n is not divisible by 4
    size_t remainder = n % 4;
    if (remainder > 0) {
        // Simple scalar path for remainder; rarely hit for large tensors
        // In practice, we suggest padding, but here we just handle it
        auto in_ptr = input.data_ptr<float>();
        auto out_ptr = output.data_ptr<float>();
        for (size_t i = n - remainder; i < n; ++i) {
            float val = in_ptr[i];
            out_ptr[i] = (val > 0.0f) ? val : (negative_slope * val);
        }
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>
void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_forward, "Vectorized Leaky ReLU (Grid-Stride)");
}
"""

# Build the extension
leaky_relu_ext = load_inline(
    name='leaky_relu_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized functional model. Uses grid-stride loops and float4 reading
    to saturate the VRAM bus of the RTX 2080Ti.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Ensure float32 (the kernel is typed for float pointer arithmetic)
    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    return output
