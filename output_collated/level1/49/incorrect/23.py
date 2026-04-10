# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_154447/code_2.py
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

# --- CUDA Kernel Code ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
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

    // Each thread processes multiple elements with a stride
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        scalar_t val = row_ptr[col];
        thread_max = max(thread_max, val);
    }

    // Warp-level reduction using shuffle
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        thread_max = max(thread_max,
                         __shfl_xor_sync(0xffffffff, thread_max, offset));

    // Write per-warp results to shared memory
    __shared__ scalar_t shmem[32]; // Max 32 warps in a block
    if (threadIdx.x % warpSize == 0)
        shmem[threadIdx.x / warpSize] = thread_max;
    __syncthreads();

    // First warp reduces the partial results from shared memory
    if (threadIdx.x < warpSize) {
        scalar_t block_max = (threadIdx.x < (blockDim.x/warpSize))
                             ? shmem[threadIdx.x]
                             : -std::numeric_limits<scalar_t>::infinity();
        for (int offset = warpSize/2; offset > 0; offset /= 2)
            block_max = max(block_max,
                            __shfl_xor_sync(0xffffffff, block_max, offset));
        if (threadIdx.x == 0)
            dst[row] = block_max;
    }
}

// Host function to launch the kernel
void max_reduce(torch::Tensor src, torch::Tensor dst) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(src.dtype() == torch::kFloat && dst.dtype() == torch::kFloat,
                "Only float tensors are supported");
    
    int rows = src.size(0) * src.size(1); // Total number of rows to reduce
    int cols = src.size(2);               // Reduction dimension size

    const int threads = 256;
    const int blocks = rows;

    AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "max_reduce", ([&] {
        max_reduce_kernel<scalar_t><<<blocks, threads>>>(
            src.data_ptr<scalar_t>(),
            dst.data_ptr<scalar_t>(),
            rows,
            cols);
    }));
}
"""

# --- C++ Binding Code ---
cpp_source = r"""
#include <torch/extension.h>

void max_reduce(torch::Tensor src, torch::Tensor dst);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduce", &max_reduce, "Max reduction kernel along last dimension");
}
"""

# --- Compile the Extension ---
max_ext = load_inline(
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
    """
    Performs max reduction along the specified dimension using a custom fused CUDA kernel.
    Optimized for dim=2 (last dimension) on large tensors.
    """
    assert dim == 2, "max_reduce kernel only supports dim=2"
    
    # Allocate output tensor with correct shape and device
    out_shape = (x.size(0), x.size(1))
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    
    # Call the custom CUDA kernel
    max_ext.max_reduce(x, out)
    
    return out

batch_size = 128
dim1 = 4096
dim2 = 4095

def get_init_inputs():
    return [1] # Example, change to desired dimension

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    return [x]
