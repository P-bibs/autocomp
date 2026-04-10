# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_154447/code_6.py
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

# --- CUDA Kernel and C++ Interface ---
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>

template <typename scalar_t>
__global__ void max_reduce_kernel(
    const scalar_t* __restrict__ src,
    scalar_t* __restrict__ dst,
    int rows,
    int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const scalar_t* row_ptr = src + row * cols;
    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();

    // Coalesced load: each thread processes a subset of columns
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        thread_max = fmaxf(thread_max, row_ptr[col]);
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, offset));

    // Shared memory for block reduction
    __shared__ scalar_t shmem[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    if (lane == 0) shmem[wid] = thread_max;
    __syncthreads();

    // First warp reduces the partials from each warp in the block
    if (threadIdx.x < warpSize) {
        scalar_t val = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shmem[threadIdx.x] : -std::numeric_limits<scalar_t>::infinity();
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
        
        if (threadIdx.x == 0)
            dst[row] = val;
    }
}

void max_reduce(torch::Tensor src, torch::Tensor dst) {
    int rows = src.size(0) * src.size(1);
    int cols = src.size(2);
    const int threads = 256;
    
    AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "max_reduce", ([&] {
        max_reduce_kernel<scalar_t><<<rows, threads>>>(
            src.data_ptr<scalar_t>(),
            dst.data_ptr<scalar_t>(),
            rows,
            cols);
    }));
}
"""

cpp_source = r"""
void max_reduce(torch::Tensor src, torch::Tensor dst);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduce", &max_reduce, "Max reduction along last dim");
}
"""

# Compile the extension just-in-time
max_ext = load_inline(
    name='max_reduce_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# --- Original Setup ---
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_init_inputs():
    return [1]

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def functional_model(x, *, dim):
    """
    Optimized functional_model using a fused CUDA kernel for reduction.
    """
    assert dim == 2, "This custom kernel only supports reduction on the last dimension."
    
    # Pre-allocate output tensor
    out = torch.empty(x.size(0), x.size(1), device=x.device, dtype=x.dtype)
    
    # Call the JIT-compiled C++ extension
    max_ext.max_reduce(x, out)
    
    return out
