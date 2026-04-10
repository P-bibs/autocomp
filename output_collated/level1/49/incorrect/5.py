# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_151147/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs Max reduction over a specific dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

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
    if 'dim' in flat_state:
        state_kwargs['dim'] = flat_state['dim']
    else:
        state_kwargs['dim'] = getattr(model, 'dim')
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for max reduction with shared memory optimization
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void max_reduction_kernel(
    const float* input,
    float* output,
    const int batch_size,
    const int dim1,
    const int dim2
) {
    extern __shared__ float shared_data[];
    
    int batch_idx = blockIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y;
    int thread_x = threadIdx.x;
    
    if (batch_idx >= batch_size || row_idx >= dim1) return;
    
    // Each thread processes multiple elements in the reduced dimension
    float thread_max = -INFINITY;
    int elements_per_thread = (dim2 + blockDim.x - 1) / blockDim.x;
    int start_idx = thread_x * elements_per_thread;
    int end_idx = min(start_idx + elements_per_thread, dim2);
    
    // Load data into registers and compute local max
    for (int i = start_idx; i < end_idx; i++) {
        int input_idx = batch_idx * dim1 * dim2 + row_idx * dim2 + i;
        thread_max = fmaxf(thread_max, input[input_idx]);
    }
    
    // Store local max in shared memory
    shared_data[thread_x * blockDim.y + tid] = thread_max;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_x < stride) {
            shared_data[thread_x * blockDim.y + tid] = fmaxf(
                shared_data[thread_x * blockDim.y + tid],
                shared_data[(thread_x + stride) * blockDim.y + tid]
            );
        }
        __syncthreads();
    }
    
    // First thread in x-dimension writes the result for this y-thread
    if (thread_x == 0) {
        shared_data[tid] = shared_data[tid];
    }
    __syncthreads();
    
    // Now reduce along the y-dimension
    for (int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    
    // Write final result
    if (tid == 0 && thread_x == 0) {
        int output_idx = batch_idx * dim1 + row_idx;
        output[output_idx] = shared_data[0];
    }
}

void max_reduction_forward(
    const at::Tensor& input,
    at::Tensor& output,
    const int batch_size,
    const int dim1,
    const int dim2
) {
    // Use 32x32 thread blocks for optimal memory access pattern
    dim3 block(32, 32);
    dim3 grid(batch_size, (dim1 + block.y - 1) / block.y);
    
    // Shared memory size: 32x32 floats
    size_t shared_mem_size = block.x * block.y * sizeof(float);
    
    max_reduction_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim1,
        dim2
    );
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void max_reduction_forward(
    const at::Tensor& input,
    at::Tensor& output,
    const int batch_size,
    const int dim1,
    const int dim2
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduction", &max_reduction_forward, "Max reduction with shared memory optimization");
}
"""

# Compile the extension with optimization flags
max_reduction_ext = load_inline(
    name='max_reduction_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    dim,
):
    batch_size, dim1, dim2 = x.shape
    output = torch.empty(batch_size, dim1, device=x.device, dtype=x.dtype)
    max_reduction_ext.max_reduction(x, output, batch_size, dim1, dim2)
    return output

batch_size = 128
dim1 = 4096
dim2 = 4095

def get_init_inputs():
    return [1] # Example, change to desired dimension

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    return [x]
