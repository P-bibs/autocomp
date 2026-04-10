# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150540/code_7.py
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

# ----------------------------------------------------------------------
# CUDA kernel + binding
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

// Each thread block processes one independent reduction (one row)
__global__ void max_reduce_kernel(
    const float* __restrict__ input,
    float* output,
    int reduce_dim,
    int other_size)
{
    // Shared memory for tree-based reduction
    extern __shared__ float sdata[];

    int idx = blockIdx.x;
    if (idx >= other_size) return;

    const float* row = input + idx * (long long)reduce_dim;

    // Per-thread partial max
    float local_max = -1e38f; 
    for (int i = threadIdx.x; i < reduce_dim; i += blockDim.x) {
        local_max = fmaxf(local_max, row[i]);
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    // Tree-based reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[idx] = sdata[0];
    }
}

void max_reduce(const torch::Tensor input,
                const int reduce_dim,
                const int other_size,
                torch::Tensor output)
{
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    const int threads = 128;
    const int blocks = other_size;
    const size_t smem = threads * sizeof(float);

    max_reduce_kernel<<<blocks, threads, smem>>>(in_ptr, out_ptr, reduce_dim, other_size);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void max_reduce(const torch::Tensor input,
                const int reduce_dim,
                const int other_size,
                torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduce", &max_reduce, "Max reduction using shared-memory kernel");
}
"""

# Compile the extension
max_ext = load_inline(
    name='max_reduce',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

def functional_model(x: torch.Tensor, *, dim: int) -> torch.Tensor:
    """
    Optimized functional_model using a custom CUDA kernel for max reduction.
    """
    if not x.is_cuda:
        x = x.cuda()

    # Move reduction axis to last position
    x = torch.moveaxis(x, dim, -1)
    
    # Ensure memory is contiguous for the kernel
    if not x.is_contiguous():
        x = x.contiguous()
        
    shape = x.shape
    reduce_dim = shape[-1]
    other_shape = shape[:-1]
    other_size = other_shape.numel()

    # Prepare input as 2D for kernel
    x_flat = x.view(other_size, reduce_dim)
    # Target output
    out = torch.empty(other_shape, dtype=x.dtype, device=x.device)

    # Launch kernel
    max_ext.max_reduce(x_flat, reduce_dim, other_size, out)
    
    return out

# ----------------------------------------------------------------------
# Helpers for evaluation harness
# ----------------------------------------------------------------------
def get_init_inputs():
    return [1]

def get_inputs():
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    # Using float32 as expected by the kernel
    x = torch.rand(batch_size, dim1, dim2, dtype=torch.float32)
    return [x]
