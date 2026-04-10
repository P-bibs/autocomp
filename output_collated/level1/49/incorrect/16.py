# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152339/code_6.py
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

# -----------------------------------------------------------------------------
# Tiled Max Reduction Kernel
# -----------------------------------------------------------------------------
# The kernel targets the reduction of the innermost dimension (dim=2).
# Strategy:
# 1. Each thread block handles one vector of size `reduce_len`.
# 2. Each thread loads elements in a coalesced fashion, computing a partial max.
# 3. Partial results are stored in shared memory and reduced using a binary tree.
# -----------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

#define TILE_SIZE 512

__global__ void max_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int num_slices,
    const int reduce_len)
{
    // Each block handles one slice
    const int slice_idx = blockIdx.x;
    if (slice_idx >= num_slices) return;

    __shared__ float sdata[TILE_SIZE];

    const float* slice_ptr = input + (slice_idx * reduce_len);
    
    // Per-thread maximum initialized to lowest possible float
    float thread_max = -FLT_MAX;

    // Strided loop to cover the entire reduction dimension
    for (int i = threadIdx.x; i < reduce_len; i += blockDim.x) {
        thread_max = fmaxf(thread_max, slice_ptr[i]);
    }

    sdata[threadIdx.x] = thread_max;
    __syncthreads();

    // Binary tree reduction in shared memory
    for (int s = TILE_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[slice_idx] = sdata[0];
    }
}

void max_reduce_launcher(torch::Tensor input, torch::Tensor output) {
    const int num_slices = input.numel() / input.size(-1);
    const int reduce_len = input.size(-1);
    
    const int threads = TILE_SIZE;
    const int blocks = num_slices;

    max_reduce_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_slices,
        reduce_len
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void max_reduce_launcher(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduce", &max_reduce_launcher, "Tiled max reduction kernel");
}
"""

# Compile the extension
tiled_ext = load_inline(
    name="tiled_reduction",
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False
)

def functional_model(x, *, dim):
    """
    Computes max along dim=2 on GPU using a tiled kernel.
    """
    if not x.is_cuda:
        return torch.max(x, dim=dim)[0]
    
    # Ensure contiguous for the kernel's memory access pattern
    x = x.contiguous()
    
    # Output shape: slice away the reduction dimension
    out_shape = list(x.shape)
    out_shape.pop(dim)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    tiled_ext.max_reduce(x, out)
    return out

# -----------------------------------------------------------------------------
# API compliance
# -----------------------------------------------------------------------------
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_init_inputs():
    return [1]

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device="cuda")
    return [x]
