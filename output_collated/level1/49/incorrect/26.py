# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_154447/code_7.py
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
# CUDA Source: Optimized Max Reduction Kernels
# -----------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

// Reduction kernel using shared memory for block-level synchronization
// This template handles different dimensions by adjusting how threadIdx 
// and indices are mapped, ensuring coalesced access where possible.

__global__ void max_reducer_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   const int outer, const int reduction_dim, const int inner,
                                   const int dim_idx)
{
    // out_idx is the index into the output tensor (size = outer * inner)
    int out_idx = blockIdx.x;
    int b = out_idx / inner;
    int i = out_idx % inner;

    float local_max = -FLT_MAX;

    // Strided loop for reduction dimension
    for (int r = threadIdx.x; r < reduction_dim; r += blockDim.x) {
        int idx;
        if (dim_idx == 0) idx = (r * (int)gridDim.y + b) * inner + i;
        else if (dim_idx == 1) idx = (b * (int)gridDim.y + r) * inner + i;
        else idx = (b * (int)gridDim.y + i) * reduction_dim + r;
        
        float val = input[idx];
        if (val > local_max) local_max = val;
    }

    extern __shared__ float sdata[];
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    // Tree reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x + s] > sdata[threadIdx.x])
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) output[out_idx] = sdata[0];
}

void max_dim_cuda(torch::Tensor input, int dim, torch::Tensor output) {
    const int B = input.size(0);
    const int N = input.size(1);
    const int M = input.size(2);
    
    int outer, red, inner;
    if (dim == 0) { outer = N; red = B; inner = M; }
    else if (dim == 1) { outer = B; red = N; inner = M; }
    else { outer = B; red = M; inner = N; }

    const int threads = 256;
    const int blocks = outer * inner;
    
    // gridDim.y holds N to help index calculation inside kernel
    max_reducer_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), outer, red, inner, dim
    );
}
"""

# -----------------------------------------------------------------------------
# C++ Interface
# -----------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void max_dim_cuda(torch::Tensor input, int dim, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_dim_cuda", &max_dim_cuda, "Max reduction along dimension");
}
"""

# -----------------------------------------------------------------------------
# Compilation
# -----------------------------------------------------------------------------
max_ext = load_inline(
    name="max_cuda_optimized",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)

def functional_model(x, *, dim):
    """
    Optimized max reduction using hand-written CUDA kernel.
    """
    if not x.is_cuda:
        x = x.cuda()
    
    # Ensure contiguous for predictable memory indexing
    x = x.contiguous()
    
    shape = list(x.shape)
    reduction_dim_size = shape.pop(dim)
    output = torch.empty(shape, dtype=x.dtype, device=x.device)
    
    max_ext.max_dim_cuda(x, dim, output)
    return output
