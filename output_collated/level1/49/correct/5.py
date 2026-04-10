# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150540/code_0.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <climits>
#include <cfloat>

__global__ void max_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int outer_size,
    const int reduction_size,
    const int inner_size
) {
    // Calculate global indices
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Handle cases where we have more blocks than output elements
    if (bid >= outer_size * inner_size) return;
    
    // Calculate which output element this block is responsible for
    int outer_idx = bid / inner_size;
    int inner_idx = bid % inner_size;
    
    // Shared memory for block-level reduction
    extern __shared__ float sdata[];
    
    // Each thread loads and processes multiple elements
    float thread_max = -FLT_MAX;
    
    // Grid-stride loop for processing elements
    for (int i = tid; i < reduction_size; i += blockDim.x) {
        int input_idx = outer_idx * reduction_size * inner_size + 
                        i * inner_size + 
                        inner_idx;
        thread_max = fmaxf(thread_max, input[input_idx]);
    }
    
    // Store in shared memory
    sdata[tid] = thread_max;
    __syncthreads();
    
    // Block-level reduction using shared memory
    for (int s = blockDim.x / 2; s > 16; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Warp-level reduction (no sync needed within a warp)
    if (tid < 16) {
        volatile float* vsdata = sdata;
        vsdata[tid] = fmaxf(vsdata[tid], vsdata[tid + 16]);
        vsdata[tid] = fmaxf(vsdata[tid], vsdata[tid + 8]);
        vsdata[tid] = fmaxf(vsdata[tid], vsdata[tid + 4]);
        vsdata[tid] = fmaxf(vsdata[tid], vsdata[tid + 2]);
        vsdata[tid] = fmaxf(vsdata[tid], vsdata[tid + 1]);
    }
    
    // Thread 0 writes the result
    if (tid == 0) {
        int output_idx = outer_idx * inner_size + inner_idx;
        output[output_idx] = sdata[0];
    }
}

void max_reduction_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    const int dim
) {
    // Calculate dimensions
    int outer_size = 1;
    int reduction_size = input.size(dim);
    int inner_size = 1;
    
    // Calculate outer and inner dimensions
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int blocks = outer_size * inner_size;
    const int shared_mem_size = threads_per_block * sizeof(float);
    
    // Launch kernel
    max_reduction_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        outer_size,
        reduction_size,
        inner_size
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void max_reduction_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    const int dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduction", &max_reduction_forward, "Max reduction along dimension");
}
"""

# Compile the extension
max_reduce_ext = load_inline(
    name='max_reduce_ext',
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
    # Calculate output shape by removing the reduction dimension
    output_shape = list(x.shape)
    output_shape.pop(dim)
    
    # Create output tensor
    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    
    # Call custom CUDA kernel
    max_reduce_ext.max_reduction(x, output, dim)
    
    return output

batch_size = 128
dim1 = 4096
dim2 = 4095

def get_init_inputs():
    return [2] # dim=2 for reduction along last dimension

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    return [x]
