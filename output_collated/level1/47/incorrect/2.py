# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_125421/code_0.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256

__global__ void sum_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2
) {
    int batch_idx = blockIdx.x;
    int row_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || row_idx >= dim1) return;
    
    // Shared memory for reduction
    __shared__ float sdata[BLOCK_SIZE];
    
    float sum = 0.0f;
    
    // Each thread sums elements with stride BLOCK_SIZE
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        sum += input[batch_idx * dim1 * dim2 + row_idx * dim2 + i];
    }
    
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (threadIdx.x == 0) {
        output[batch_idx * dim1 + row_idx] = sdata[0];
    }
}

void sum_reduce_forward(
    const at::Tensor input,
    at::Tensor output,
    const int batch_size,
    const int dim1,
    const int dim2
) {
    const int threads_per_block = BLOCK_SIZE;
    const dim3 blocks(batch_size, dim1);
    
    sum_reduce_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim1,
        dim2
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function in the .cu file
void sum_reduce_forward(
    const at::Tensor input,
    at::Tensor output,
    const int batch_size,
    const int dim1,
    const int dim2
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_reduce", &sum_reduce_forward, "Sum reduce kernel");
}
"""

# Compile the extension
sum_reduce_ext = load_inline(
    name='sum_reduce',
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
    # Create output tensor with correct shape (keepdim=True)
    output = torch.zeros(batch_size, dim1, 1, device=x.device, dtype=x.dtype)
    
    # Use custom CUDA kernel for reduction
    sum_reduce_ext.sum_reduce(x, output, batch_size, dim1, dim2)
    
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

# Move model to CUDA for evaluation
if __name__ == "__main__":
    # Test the function
    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    result = functional_model(x, dim=reduce_dim)
    print(f"Output shape: {result.shape}")
    
    # Verify correctness against PyTorch
    expected = torch.sum(x, dim=reduce_dim, keepdim=True)
    print(f"Max difference: {torch.max(torch.abs(result - expected))}")
