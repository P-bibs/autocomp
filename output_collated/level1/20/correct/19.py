# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_202725/code_20.py
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

# The CUDA kernel uses a Grid-Stride loop to decouple grid dimension from grid size.
# This ensures that even for massive buffers, we achieve high occupancy and utilize
# the GPU's scheduler to hide memory latency. We use float4 vectorization for 
# coalesced memory access (128-bit loads/stores).
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void leaky_relu_grid_stride_kernel(const float* __restrict__ input, float* __restrict__ output, float negative_slope, int numel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // We increment by stride * 4 because each thread processes 4 elements (1 float4) per iteration.
    for (int i = tid * 4; i < numel; i += stride * 4) {
        if (i + 3 < numel) {
            float4 in_vec = reinterpret_cast<const float4*>(input + i)[0];
            float4 out_vec;
            
            // Branchless implementation for performance
            out_vec.x = (in_vec.x > 0) ? in_vec.x : (in_vec.x * negative_slope);
            out_vec.y = (in_vec.y > 0) ? in_vec.y : (in_vec.y * negative_slope);
            out_vec.z = (in_vec.z > 0) ? in_vec.z : (in_vec.z * negative_slope);
            out_vec.w = (in_vec.w > 0) ? in_vec.w : (in_vec.w * negative_slope);
            
            reinterpret_cast<float4*>(output + i)[0] = out_vec;
        } else {
            // Sequential cleanup for the remainder of the array
            for (int j = i; j < numel; ++j) {
                float val = input[j];
                output[j] = (val > 0) ? val : (val * negative_slope);
            }
        }
    }
}

void launch_leaky_relu(int numel, float* input, float* output, float negative_slope) {
    const int block_size = 256;
    // 1024 blocks provides sufficient occupancy for the 2080Ti to handle latency
    const int grid_size = 1024; 
    leaky_relu_grid_stride_kernel<<<grid_size, block_size>>>(input, output, negative_slope, numel);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_leaky_relu(int numel, float* input, float* output, float negative_slope);

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    launch_leaky_relu(input.numel(), input.data_ptr<float>(), output.data_ptr<float>(), negative_slope);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu_forward", &leaky_relu_forward, "Grid-stride Vectorized Leaky ReLU");
}
"""

# Compile the extension just-in-time
leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized Leaky ReLU using a Grid-Stride Vectorized CUDA kernel.
    """
    # Ensure input is contiguous for float4 coalesced access
    if not x.is_contiguous():
        x = x.contiguous()
    
    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu_forward(x, output, float(negative_slope))
    return output
