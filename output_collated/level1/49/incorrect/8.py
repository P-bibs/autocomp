# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_151147/code_7.py
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

# -------------------------------------------------------------------------
# Inline CUDA kernel: Max Reduction
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void max_reduce_kernel(const float* __restrict__ input,
                                   float* output,
                                   const int row_size,
                                   const int num_rows) {
    // Shared memory for per-warp results (8 warps for 256 threads)
    __shared__ float sdata[8];

    int tid = threadIdx.x;
    int row = blockIdx.x;
    
    if (row >= num_rows) return;

    const float* row_ptr = input + (size_t)row * row_size;

    // Grid-stride loop: each thread processes row_size / blockDim.x elements
    float val = -FLT_MAX;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float v = row_ptr[i];
        if (v > val) val = v;
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    // Write warp results to shared memory
    if (tid % warpSize == 0) {
        sdata[tid / warpSize] = val;
    }
    __syncthreads();

    // Final reduction across warps
    if (tid < 8) {
        val = sdata[tid];
        // Reduce the 8 warp results
        for (int offset = 4; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0x000000FF, val, offset));
        }
    }

    if (tid == 0) {
        output[row] = val;
    }
}

void max_reduce(const torch::Tensor &input, torch::Tensor &output) {
    int num_rows = input.size(0);
    int row_size = input.size(1);
    const int threads = 256;
    
    max_reduce_kernel<<<num_rows, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        row_size,
        num_rows
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_reduce(const torch::Tensor &input, torch::Tensor &output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduce", &max_reduce, "max reduction along last dimension");
}
"""

# Compile extension
fused_ext = load_inline(
    name='max_reduce_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

def functional_model(x, *, dim):
    # Ensure contiguous float32
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    
    ndim = x.dim()
    if dim < 0:
        dim = ndim + dim
    
    # Permute if necessary
    if dim != ndim - 1:
        dims = list(range(ndim))
        dims[dim], dims[-1] = dims[-1], dims[dim]
        x = x.permute(dims)
    
    x = x.contiguous()
    
    # Flatten
    shape = x.shape
    M = shape[-1]
    N = x.numel() // M
    x_view = x.view(N, M)
    
    output = torch.empty(N, dtype=torch.float32, device=x.device)
    
    # Invoke custom kernel
    fused_ext.max_reduce(x_view, output)
    
    return output.view(shape[:-1])

def get_init_inputs():
    return [1]

def get_inputs():
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    return [torch.rand(batch_size, dim1, dim2, device='cuda')]
