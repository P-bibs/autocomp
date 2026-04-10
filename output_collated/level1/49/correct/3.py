# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_145843/code_7.py
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
# CUDA kernel: Tile-based reduction for maximum
# Each block handles one output element index. 
# We reduce along reduce_dim by iterating in tiles of (blockDim.x * tileWidth).
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

template <typename T>
__global__ void reduce_max_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int64_t shape0,
    const int64_t shape1,
    const int64_t shape2,
    const int64_t stride0,
    const int64_t stride1,
    const int64_t stride2,
    const int reduce_dim,
    const int64_t reduce_size,
    const int64_t out_size)
{
    const int tid = threadIdx.x;
    const int out_idx = blockIdx.x;
    
    if (out_idx >= out_size) return;

    // Calculate base offset in input tensor
    int64_t idx0, idx1, idx2;
    int64_t base = 0;
    int64_t stride_reduce = 0;

    if (reduce_dim == 0) {
        idx1 = out_idx / shape2;
        idx2 = out_idx % shape2;
        base = idx1 * stride1 + idx2 * stride2;
        stride_reduce = stride0;
    } else if (reduce_dim == 1) {
        idx0 = out_idx / shape2;
        idx2 = out_idx % shape2;
        base = idx0 * stride0 + idx2 * stride2;
        stride_reduce = stride1;
    } else {
        idx0 = out_idx / shape1;
        idx1 = out_idx % shape1;
        base = idx0 * stride0 + idx1 * stride1;
        stride_reduce = stride2;
    }

    constexpr int tileWidth = 4;
    T local_max = -std::numeric_limits<T>::infinity();

    // Tiling reduction
    for (int64_t tileStart = 0; tileStart < reduce_size; tileStart += blockDim.x * tileWidth) {
        #pragma unroll
        for (int j = 0; j < tileWidth; ++j) {
            int64_t r_idx = tileStart + tid * tileWidth + j;
            if (r_idx < reduce_size) {
                T val = input[base + r_idx * stride_reduce];
                if (val > local_max) local_max = val;
            }
        }
    }

    // Block-level reduction using shared memory
    extern __shared__ char sdata_raw[];
    T* sdata = reinterpret_cast<T*>(sdata_raw);
    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[out_idx] = sdata[0];
}

void fused_op_forward(torch::Tensor input, int reduce_dim, torch::Tensor output) {
    auto sizes = input.sizes();
    int64_t out_size = output.numel();
    int64_t reduce_size = input.size(reduce_dim);
    int threads = 256;
    int blocks = static_cast<int>(out_size);
    int shared_mem = threads * input.element_size();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "reduce_max_kernel", ([&] {
        reduce_max_kernel<scalar_t><<<blocks, threads, shared_mem>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            sizes[0], sizes[1], sizes[2],
            input.stride(0), input.stride(1), input.stride(2),
            reduce_dim, reduce_size, out_size
        );
    }));
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, int reduce_dim, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Tile-based max reduction");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3'],
    with_cuda=True
)

def functional_model(x: torch.Tensor, *, dim: int) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Calculate output shape
    out_shape = list(x.shape)
    out_shape.pop(dim)
    output = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    
    # Run custom kernel
    fused_ext.fused_op(x, dim, output)
    return output
