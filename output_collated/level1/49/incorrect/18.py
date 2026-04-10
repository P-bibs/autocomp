# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_153700/code_0.py
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

# CUDA kernel for efficient max reduction
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

__global__ void max_reduction_kernel(
    const float* input,
    float* output,
    const int batch_size,
    const int dim1,
    const int dim2
) {
    // Shared memory for partial results within a block
    extern __shared__ float shared_data[];
    
    int batch_idx = blockIdx.x;
    int row_idx = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Boundary check
    if (batch_idx >= batch_size || row_idx >= dim1) return;
    
    // Calculate global index base
    int input_base_idx = batch_idx * dim1 * dim2 + row_idx * dim2;
    
    // Perform reduction in grid-stride manner
    float thread_max = -FLT_MAX;
    for (int i = tid; i < dim2; i += block_size) {
        thread_max = fmaxf(thread_max, input[input_base_idx + i]);
    }
    
    // Store thread result in shared memory
    shared_data[tid] = thread_max;
    __syncthreads();
    
    // Perform reduction in shared memory using warp-level primitives for better efficiency
    for (int stride = block_size / 2; stride > 16; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    
    // Unroll the last 6 iterations (warp-level optimization)
    if (tid < 16) {
        shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + 16]);
    }
    if (tid < 8) {
        shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + 8]);
    }
    if (tid < 4) {
        shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + 4]);
    }
    if (tid < 2) {
        shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + 2]);
    }
    if (tid < 1) {
        shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + 1]);
    }
    
    // Write result
    if (tid == 0) {
        output[batch_idx * dim1 + row_idx] = shared_data[0];
    }
}

void max_reduction_forward(
    const at::Tensor input,
    at::Tensor output,
    const int batch_size,
    const int dim1,
    const int dim2
) {
    // Calculate grid and block dimensions
    const int threads_per_block = 256;
    const int shared_mem_size = threads_per_block * sizeof(float);
    
    dim3 grid(batch_size, dim1);
    dim3 block(threads_per_block);
    
    max_reduction_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim1,
        dim2
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR(cudaGetErrorString(err));
    }
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void max_reduction_forward(
    const at::Tensor input,
    at::Tensor output,
    const int batch_size,
    const int dim1,
    const int dim2
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduction", &max_reduction_forward, "Max reduction forward pass");
}
"""

# Compile the extension
max_reduction_ext = load_inline(
    name='max_reduction',
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
    output = torch.empty(batch_size, dim1, dtype=x.dtype, device=x.device)
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
