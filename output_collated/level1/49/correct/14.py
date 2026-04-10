# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_153700/code_7.py
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
# CUDA kernel: Parallel max reduction
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void reduce_max_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int outer,
    const int reduction,
    const int inner,
    const int out_size)
{
    const int idx = blockIdx.x;
    if (idx >= out_size) return;

    const int outer_idx = idx / inner;
    const int inner_idx = idx % inner;
    const float* slice = input + (outer_idx * reduction * inner) + inner_idx;

    // Use shared memory for partial reductions
    extern __shared__ float sdata[];
    
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < reduction; i += blockDim.x) {
        float val = __ldg(&slice[i * inner]);
        if (val > max_val) max_val = val;
    }

    sdata[threadIdx.x] = max_val;
    __syncthreads();

    // Block-level reduction
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x + s] > sdata[threadIdx.x]) {
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    // Warp-level reduction
    if (threadIdx.x < 32) {
        float val = sdata[threadIdx.x];
        if (threadIdx.x + 32 < blockDim.x) {
            float other = sdata[threadIdx.x + 32];
            if (other > val) val = other;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffff, val, offset);
            if (other > val) val = other;
        }
        if (threadIdx.x == 0) output[idx] = val;
    }
}

void fused_max_cuda(torch::Tensor input, torch::Tensor output, int64_t dim) {
    auto shape = input.sizes();
    int ndim = shape.size();
    int outer = 1;
    for(int i=0; i<dim; ++i) outer *= shape[i];
    int reduction = shape[dim];
    int inner = 1;
    for(int i=dim+1; i<ndim; ++i) inner *= shape[i];

    const int block_size = 256;
    const int grid = outer * inner;
    
    reduce_max_kernel<<<grid, block_size, block_size * sizeof(float)>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        outer, reduction, inner, grid
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_max_cuda(torch::Tensor input, torch::Tensor output, int64_t dim);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_max", &fused_max_cuda, "Fused max reduction kernel");
}
"""

# Compile the inline extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    # Ensure input is standard ready
    x = x.contiguous().cuda().float()
    
    # Calculate output shape
    shape = list(x.shape)
    reduced_shape = shape[:dim] + shape[dim+1:]
    output = torch.empty(reduced_shape, device=x.device, dtype=x.dtype)
    
    # Execute kernel
    fused_ext.fused_max(x, output, dim)
    return output

# Helper variables for evaluation
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_init_inputs():
    return [1]

def get_inputs():
    return [torch.rand(batch_size, dim1, dim2)]
