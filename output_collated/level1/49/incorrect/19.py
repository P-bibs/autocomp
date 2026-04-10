# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_153700/code_2.py
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
from torch.utils.cpp_extension import load_inline

# CUDA kernel for max reduction over the last dimension
cuda_kernel = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__global__ void max_reduce_last_dim_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int reduction_size
) {
    // Each block handles one output element
    const int output_idx = blockIdx.x;
    const int stride = reduction_size;
    
    const scalar_t* row = input + output_idx * stride;
    
    // Per-thread reduction with grid-stride loop
    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();
    for (int i = threadIdx.x; i < reduction_size; i += blockDim.x) {
        scalar_t val = row[i];
        thread_max = fmax(thread_max, val);
    }
    
    // Warp-level reduction using shuffle
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        scalar_t n = __shfl_down_sync(0xffffffff, thread_max, offset);
        thread_max = fmax(thread_max, n);
    }
    
    // Store warp results in shared memory
    __shared__ scalar_t warp_max[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    if (lane_id == 0) {
        warp_max[warp_id] = thread_max;
    }
    __syncthreads();
    
    // Final reduction of warp results by first warp
    if (warp_id == 0) {
        scalar_t warp_val = (lane_id < (blockDim.x + 31) / 32) ? 
                            warp_max[lane_id] : 
                            -std::numeric_limits<scalar_t>::infinity();
        
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            scalar_t n = __shfl_down_sync(0xffffffff, warp_val, offset);
            warp_val = fmax(warp_val, n);
        }
        
        if (lane_id == 0) {
            output[output_idx] = warp_val;
        }
    }
}

void max_reduce_last_dim(
    torch::Tensor input,
    torch::Tensor output,
    const int reduction_size
) {
    const int total_outputs = input.size(0) * input.size(1);
    const int threads_per_block = 256;
    const int blocks = total_outputs;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_reduce_last_dim", [&] {
        max_reduce_last_dim_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduction_size
        );
    });
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR(cudaGetErrorString(err));
    }
}
'''

# C++ binding
cpp_source = r'''
#include <torch/extension.h>

void max_reduce_last_dim(
    torch::Tensor input,
    torch::Tensor output,
    const int reduction_size
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduce_last_dim", &max_reduce_last_dim, "Max reduction over last dimension");
}
'''

# Compile the extension
custom_max_ext = load_inline(
    name='custom_max_reduce',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Custom CUDA implementation of torch.max(x, dim=dim)[0] for last dimension reduction"""
    if not x.is_cuda:
        raise RuntimeError("Input tensor must be on CUDA device")
    if dim != x.dim() - 1:
        raise NotImplementedError("Only reduction over last dimension is supported")
    
    # Create output tensor with reduced dimension removed
    output_shape = list(x.shape)
    output_shape.pop(dim)
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Launch custom kernel
    custom_max_ext.max_reduce_last_dim(x, output, x.size(dim))
    
    return output

# Constants (unchanged)
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_init_inputs():
    return [1]

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    return [x]
