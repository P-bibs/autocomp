# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_153700/code_4.py
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

# CUDA kernel for efficient max reduction
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void max_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int num_rows,
    const int dim2
) {
    // Shared memory for partial results of one row
    // Each block processes one row (dim1 index)
    extern __shared__ float shared_data[];
    
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // Each block handles one row of the input
    const float* row_data = input + (size_t)row_idx * dim2;
    
    float thread_max = -FLT_MAX;
    
    // Grid-stride loop to handle cases where dim2 > blockDim.x
    for (int i = tid; i < dim2; i += blockDim.x) {
        thread_max = fmaxf(thread_max, row_data[i]);
    }
    
    shared_data[tid] = thread_max;
    __syncthreads();
    
    // Perform tree-based reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[row_idx] = shared_data[0];
    }
}

void max_reduction_forward(
    const at::Tensor input,
    at::Tensor output
) {
    const int batch_size = input.size(0);
    const int dim1 = input.size(1);
    const int dim2 = input.size(2);
    const int num_rows = batch_size * dim1;
    
    // Use 256 threads per block for good occupancy
    const int threads = 256;
    const int shared_mem_size = threads * sizeof(float);
    
    max_reduction_kernel<<<num_rows, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_rows,
        dim2
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void max_reduction_forward(const at::Tensor input, at::Tensor output);

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

def functional_model(x, *, dim):
    # Only dim=2 is supported based on the original logic
    assert dim == 2 or dim == -1, "Custom kernel only supports reduction on the last dimension"
    
    batch_size, dim1, dim2 = x.shape
    # Flatten batch and dim1 into a single dimension for the kernel
    output = torch.empty(batch_size, dim1, dtype=x.dtype, device=x.device)
    
    # Reshape input to (N, dim2) where N = batch_size * dim1
    # We create a view to satisfy the kernel expectation
    x_view = x.view(-1, dim2)
    output_view = output.view(-1)
    
    max_reduction_ext.max_reduction(x_view, output_view)
    return output

# Parameters
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_init_inputs():
    return [1]

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    return [x]
