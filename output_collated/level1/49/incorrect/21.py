# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_153700/code_6.py
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

# ---------- CUDA kernel (max reduction over the last dimension) ----------
# Optimized with grid-stride loops and warp-level reduction
cuda_kernel = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>

template <typename scalar_t>
__global__ void max_reduce_last_dim_kernel(
        const scalar_t* __restrict__ in,
        scalar_t* __restrict__ out,
        const int total_rows,
        const int dim2)
{
    // One block processes one output element (row)
    const int row_idx = blockIdx.x;
    if (row_idx >= total_rows) return;

    const scalar_t* row_ptr = in + (size_t)row_idx * dim2;
    
    // Per-thread reduction using a grid-stride loop
    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        thread_max = max(thread_max, row_ptr[i]);
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        scalar_t other = __shfl_xor_sync(0xffffffff, thread_max, offset);
        thread_max = max(thread_max, other);
    }

    // Shared memory for block-wide reduction
    __shared__ scalar_t s_data[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    if (lane == 0) s_data[wid] = thread_max;
    __syncthreads();

    // Final reduction of the warps
    if (wid == 0) {
        scalar_t w_max = (threadIdx.x < (blockDim.x / warpSize)) ? s_data[lane] : -std::numeric_limits<scalar_t>::infinity();
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            scalar_t other = __shfl_xor_sync(0xffffffff, w_max, offset);
            w_max = max(w_max, other);
        }
        if (threadIdx.x == 0) {
            out[row_idx] = w_max;
        }
    }
}

void max_reduce(torch::Tensor input, torch::Tensor output, int dim2) {
    const int total_rows = input.numel() / dim2;
    const int threads = 256;
    const int blocks = total_rows;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_reduce_kernel", ([&] {
        max_reduce_last_dim_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            total_rows,
            dim2
        );
    }));
}
'''

# ---------- C++ binding ----------
cpp_source = r'''
#include <torch/extension.h>
void max_reduce(torch::Tensor input, torch::Tensor output, int dim2);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduce", &max_reduce, "Max reduction over last dimension");
}
'''

# Compile on-the-fly
_ext = load_inline(
    name='custom_max_reduce',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

def functional_model(x, *, dim):
    """
    Optimized functional_model using custom CUDA max reduction.
    """
    # Force flattening to reduce last dim if necessary, 
    # but based on requirements, we assume input is [B, D1, D2]
    # and we reduce on dim 2.
    
    # Calculate output shape
    shape = list(x.shape)
    reduced_dim = shape.pop(dim)
    output = torch.empty(shape, device=x.device, dtype=x.dtype)
    
    # Ensure input is contiguous for coalesced access
    x_contig = x.contiguous()
    
    _ext.max_reduce(x_contig, output, reduced_dim)
    return output

# --- Metadata for test harness ---
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_init_inputs():
    return [1]

def get_inputs():
    return [torch.rand(batch_size, dim1, dim2, device='cuda')]
