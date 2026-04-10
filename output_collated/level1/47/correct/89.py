# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_143434/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs sum reduction over a specified dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
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

# CUDA kernel for fused operation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2,
    const int reduce_dim
) {
    extern __shared__ float shared_data[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    if (reduce_dim == 1) {
        // Reduce along dim1 (middle dimension)
        int batch_idx = blockIdx.y;
        int dim2_idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (batch_idx < batch_size && dim2_idx < dim2) {
            float sum = 0.0f;
            
            // Each thread sums across dim1 for its (batch, dim2) position
            for (int i = 0; i < dim1; i++) {
                int input_idx = batch_idx * dim1 * dim2 + i * dim2 + dim2_idx;
                sum += input[input_idx];
            }
            
            int output_idx = batch_idx * dim2 + dim2_idx;
            output[output_idx] = sum;
        }
    } else if (reduce_dim == 2) {
        // Reduce along dim2 (last dimension)
        int batch_idx = blockIdx.y;
        int dim1_idx = blockIdx.x;
        
        if (batch_idx < batch_size && dim1_idx < dim1) {
            float sum = 0.0f;
            
            // Each thread block handles one dim1 element per batch
            for (int i = tid; i < dim2; i += blockDim.x) {
                int input_idx = batch_idx * dim1 * dim2 + dim1_idx * dim2 + i;
                sum += input[input_idx];
            }
            
            // Store in shared memory
            shared_data[tid] = sum;
            __syncthreads();
            
            // Reduction in shared memory
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared_data[tid] += shared_data[tid + s];
                }
                __syncthreads();
            }
            
            // Write result
            if (tid == 0) {
                int output_idx = batch_idx * dim1 + dim1_idx;
                output[output_idx] = shared_data[0];
            }
        }
    }
}

void sum_reduce_forward(
    const at::Tensor input,
    at::Tensor output,
    const int reduce_dim
) {
    const int batch_size = input.size(0);
    const int dim1 = input.size(1);
    const int dim2 = input.size(2);
    
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    if (reduce_dim == 1) {
        // Reduce along dim1
        const int threads_per_block = 512;
        const int num_blocks_dim2 = (dim2 + threads_per_block - 1) / threads_per_block;
        const dim3 grid_dim(num_blocks_dim2, batch_size, 1);
        const dim3 block_dim(threads_per_block, 1, 1);
        
        sum_reduce_kernel<<<grid_dim, block_dim>>>(
            input_ptr,
            output_ptr,
            batch_size,
            dim1,
            dim2,
            reduce_dim
        );
    } else if (reduce_dim == 2) {
        // Reduce along dim2
        const int threads_per_block = 512;
        const dim3 grid_dim(dim1, batch_size, 1);
        const dim3 block_dim(threads_per_block, 1, 1);
        const int shared_mem_size = threads_per_block * sizeof(float);
        
        sum_reduce_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
            input_ptr,
            output_ptr,
            batch_size,
            dim1,
            dim2,
            reduce_dim
        );
    }
}
"""

# C++ interface bindings
cpp_source = r"""
#include <torch/extension.h>

void sum_reduce_forward(
    const at::Tensor input,
    at::Tensor output,
    const int reduce_dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_reduce", &sum_reduce_forward, "Sum reduction kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_sum_op',
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
    if dim == 1:  # Reduce along middle dimension
        output = torch.empty(batch_size, 1, dim2, dtype=x.dtype, device=x.device)
    elif dim == 2:  # Reduce along last dimension
        output = torch.empty(batch_size, dim1, 1, dtype=x.dtype, device=x.device)
    else:
        raise ValueError("Only dim=1 or dim=2 supported")
    
    fused_ext.sum_reduce(x, output, dim)
    return output

batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_init_inputs():
    return [reduce_dim]

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    return [x]
