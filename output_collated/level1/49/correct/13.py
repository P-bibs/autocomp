# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_153700/code_3.py
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
# CUDA kernel + host code (fused max)
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------
// Parallel max reduction kernel
// ---------------------------------------------------------------
__global__ void reduce_max_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int outer,
    const int reduction,
    const int inner,
    const int out_size)
{
    // index of the output element this block is responsible for
    const int idx = blockIdx.x;
    if (idx >= out_size) return;

    const int outer_idx = idx / inner;
    const int inner_idx = idx % inner;

    // pointer to the start of the slice we have to reduce
    const float* slice = input + (outer_idx * reduction * inner) + inner_idx;

    // ----- per‑thread partial max -----
    float max_val = -1e38f;                       // -FLT_MAX
    for (int i = threadIdx.x; i < reduction; i += blockDim.x) {
        // read through the read‑only cache
        float val = __ldg(&slice[i * inner]);
        if (val > max_val) max_val = val;
    }

    // ----- store per‑thread result in shared memory -----
    __shared__ float sdata[256];
    sdata[threadIdx.x] = max_val;
    __syncthreads();

    // ----- block‑level tree reduction (except the last warp) -----
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            float other = sdata[threadIdx.x + s];
            if (other > sdata[threadIdx.x]) sdata[threadIdx.x] = other;
        }
        __syncthreads();
    }

    // ----- final warp‑level reduction using shuffle -----
    if (threadIdx.x < 32) {
        float val = sdata[threadIdx.x];
        // combine the two halves of the block (if blockDim > 32)
        if (threadIdx.x + 32 < blockDim.x) {
            float other = sdata[threadIdx.x + 32];
            if (other > val) val = other;
        }
        // warp‑shuffle reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffff, val, offset);
            if (other > val) val = other;
        }
        // store result
        if (threadIdx.x == 0) output[idx] = val;
    }
}

// ---------------------------------------------------------------
// Host wrapper that sets up the launch parameters
// ---------------------------------------------------------------
torch::Tensor fused_max(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto shape = input.sizes();
    int ndim = static_cast<int>(shape.size());
    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    // sizes of the three logical parts
    int outer = 1, reduction = 1, inner = 1;
    for (int i = 0; i < dim; ++i) outer *= static_cast<int>(shape[i]);
    reduction = static_cast<int>(shape[dim]);
    for (int i = dim + 1; i < ndim; ++i) inner *= static_cast<int>(shape[i]);

    const int out_size = outer * inner;

    // build output shape (remove the reduced dimension)
    std::vector<int64_t> out_shape;
    out_shape.reserve(ndim - 1);
    for (int i = 0; i < ndim; ++i) {
        if (i != dim) out_shape.push_back(shape[i]);
    }

    auto output = torch::empty(out_shape, input.options());

    const int block_size = 256;
    const int grid = out_size;                     // one block per output element

    reduce_max_kernel<<<grid, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        outer,
        reduction,
        inner,
        out_size
    );

    return output;
}
"""

# ----------------------------------------------------------------------
# C++ binding (pybind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

torch::Tensor fused_max(torch::Tensor input, int64_t dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_max", &fused_max,
          "Fused max reduction along a given dimension using a custom CUDA kernel");
}
"""

# ----------------------------------------------------------------------
# Compile the inline extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# The functional model that will be evaluated
# ----------------------------------------------------------------------
def functional_model(x, *, dim):
    """
    Returns the max of tensor `x` along dimension `dim`.
    The implementation uses a custom CUDA kernel that exploits
    warp‑level primitives for fast reduction.
    """
    # Make sure the tensor is on the GPU and contiguous
    if not x.is_cuda:
        x = x.cuda()
    if not x.is_contiguous():
        x = x.contiguous()

    # The kernel only handles float32; cast if necessary
    if x.dtype != torch.float32:
        x = x.float()

    # Call the compiled kernel
    return fused_ext.fused_max(x, dim)

# ----------------------------------------------------------------------
# Original helpers (kept for completeness)
# ----------------------------------------------------------------------
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_init_inputs():
    return [1]  # Example, change to desired dimension

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]
