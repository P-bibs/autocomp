# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_211408/code_9.py
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

# CUDA kernel with float4 vectorization and optimal block sizing
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void leaky_relu_vec8_kernel(const float* __restrict__ input, float* __restrict__ output, float negative_slope, int numel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 8;
    
    if (idx + 7 < numel) {
        // Process 8 elements per thread using two float4 vectorized loads/stores
        float4 in_vec1 = reinterpret_cast<const float4*>(input)[idx / 4];
        float4 in_vec2 = reinterpret_cast<const float4*>(input)[(idx + 4) / 4];
        
        float4 out_vec1, out_vec2;
        
        // Vectorized Leaky ReLU computation - unrolled for better register usage
        out_vec1.x = in_vec1.x > 0.0f ? in_vec1.x : in_vec1.x * negative_slope;
        out_vec1.y = in_vec1.y > 0.0f ? in_vec1.y : in_vec1.y * negative_slope;
        out_vec1.z = in_vec1.z > 0.0f ? in_vec1.z : in_vec1.z * negative_slope;
        out_vec1.w = in_vec1.w > 0.0f ? in_vec1.w : in_vec1.w * negative_slope;
        
        out_vec2.x = in_vec2.x > 0.0f ? in_vec2.x : in_vec2.x * negative_slope;
        out_vec2.y = in_vec2.y > 0.0f ? in_vec2.y : in_vec2.y * negative_slope;
        out_vec2.z = in_vec2.z > 0.0f ? in_vec2.z : in_vec2.z * negative_slope;
        out_vec2.w = in_vec2.w > 0.0f ? in_vec2.w : in_vec2.w * negative_slope;
        
        // Store results back using vectorized writes
        reinterpret_cast<float4*>(output)[idx / 4] = out_vec1;
        reinterpret_cast<float4*>(output)[(idx + 4) / 4] = out_vec2;
    } else {
        // Handle remaining elements that don't fit in a full 8-element chunk
        for (int i = idx; i < numel && i < idx + 8; ++i) {
            float val = input[i];
            output[i] = val > 0.0f ? val : val * negative_slope;
        }
    }
}

void launch_leaky_relu(int numel, float* input, float* output, float negative_slope) {
    const int block_size = 512;
    const int elements_per_thread = 8;
    const int threads_per_block = block_size;
    const int elements_per_block = threads_per_block * elements_per_thread;
    const int grid_size = (numel + elements_per_block - 1) / elements_per_block;
    
    leaky_relu_vec8_kernel<<<grid_size, threads_per_block>>>(input, output, negative_slope, numel);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_leaky_relu(int numel, float* input, float* output, float negative_slope);

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    launch_leaky_relu(input.numel(), input.data_ptr<float>(), output.data_ptr<float>(), negative_slope);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu_forward", &leaky_relu_forward, "Optimized vectorized Leaky ReLU forward");
}
"""

leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    # Ensure input is contiguous for efficient memory access
    if not x.is_contiguous():
        x = x.contiguous()
    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu_forward(x, output, float(negative_slope))
    return output

batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
