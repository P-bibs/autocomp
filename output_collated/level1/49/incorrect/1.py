# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150540/code_3.py
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
#include <math.h>

// Reduce max over the last dimension of a (other_size × reduce_dim) tensor.
// Each block computes the maximum of one "row".
__global__ void max_reduce_kernel(
    const float* __restrict__ input,
    float* output,
    int reduce_dim,
    int other_size)
{
    // Shared memory for block-wise partial maxima
    __shared__ float sdata[128];

    int idx = blockIdx.x;                 // one block per output element
    if (idx >= other_size) return;

    // Point to the beginning of the row we are reducing
    const float* row = input + idx * reduce_dim;

    // ---- per-thread partial max ---------------------------------------
    float local_max = -1e38f;              // -infinity
    for (int i = threadIdx.x; i < reduce_dim; i += blockDim.x) {
        float val = row[i];
        if (val > local_max) local_max = val;
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    // ---- block-wise reduction (tree) ----------------------------------
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float other = sdata[threadIdx.x + s];
            if (other > sdata[threadIdx.x]) sdata[threadIdx.x] = other;
        }
        __syncthreads();
    }

    // ---- write result -------------------------------------------------
    if (threadIdx.x == 0) {
        output[idx] = sdata[0];
    }
}

// C++ wrapper that launches the kernel
void max_reduce(const torch::Tensor input,
                const int reduce_dim,
                const int other_size,
                torch::Tensor output)
{
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    const int block = 128;                         // threads per block
    const int grid  = other_size;                  // one block per output element
    const int smem  = block * sizeof(float);       // shared memory

    max_reduce_kernel<<<grid, block, smem>>>(in_ptr, out_ptr, reduce_dim, other_size);
    cudaDeviceSynchronize();
}
"""

cpp_source = r"""
#include <torch/extension.h>

void max_reduce(const torch::Tensor input,
                const int reduce_dim,
                const int other_size,
                torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduce", &max_reduce, "max reduction using shared-memory kernel");
}
"""

# Compile the inline CUDA extension
max_ext = load_inline(
    name='max_reduce',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# ----------------------------------------------------------------------
# Wrapper that reduces over the last axis using the custom kernel
# ----------------------------------------------------------------------
def _max_reduce_cuda(x: torch.Tensor) -> torch.Tensor:
    """Reduce max over the last dimension of *x* using the shared-memory kernel."""
    shape = x.shape
    reduce_dim = shape[-1]                     # size of the dimension to reduce
    other_shape = shape[:-1]                   # leading dimensions -> output shape
    other_size = other_shape.numel()           # total number of independent reductions

    # Ensure the tensor is contiguous in (other_size, reduce_dim) layout
    x_flat = x.reshape(other_size, reduce_dim)

    # Allocate output
    out = torch.empty(other_shape, dtype=x.dtype, device=x.device)

    # Launch the custom kernel
    max_ext.max_reduce(x_flat, reduce_dim, other_size, out)
    return out

# ----------------------------------------------------------------------
# The functional model that will be imported for evaluation
# ----------------------------------------------------------------------
def functional_model(x: torch.Tensor, *, dim: int) -> torch.Tensor:
    """
    Returns the maximum of *x* along dimension *dim*.
    The implementation moves the requested axis to the innermost position,
    then uses a shared-memory CUDA kernel for the reduction.
    """
    # Make sure the input lives on the GPU
    if not x.is_cuda:
        x = x.cuda()

    # Bring the reduction axis to the last position (the kernel always reduces the last axis)
    if dim != x.dim() - 1:
        x = torch.moveaxis(x, dim, -1)

    # Perform the reduction with our custom kernel
    return _max_reduce_cuda(x)

# ----------------------------------------------------------------------
# Helpers required by the benchmarking harness
# ----------------------------------------------------------------------
def get_init_inputs():
    # The original skeleton used [1] as a placeholder; we keep it unchanged.
    return [1]

def get_inputs():
    # Same random tensor as in the original code
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    x = torch.rand(batch_size, dim1, dim2)
    return [x]
