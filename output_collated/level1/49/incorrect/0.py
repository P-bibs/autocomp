# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150540/code_2.py
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




# ==============================
#  max_reduce_shared.py
# ==============================
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
#  CUDA kernel: max‑reduction over the last dimension using shared memory
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

// ------------------------------------------------------------------
//  One thread block processes one (batch, dim1) “row”.
//  Threads cooperatively load the row into shared memory, compute the
//  maximum, and write a single float to the output.
// ------------------------------------------------------------------
template <int TPB>
__global__ void max_reduce_last_dim(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int dim2,               // size of the reduced dimension
    const int stride_batch,       // distance (in floats) between successive batches
    const int stride_dim1)        // distance (in floats) between successive dim1 elements
{
    // blockIdx.x = batch index
    // blockIdx.y = dim1 index
    const int b   = blockIdx.x;
    const int i   = blockIdx.y;

    // start of the row to be reduced
    const float* row_ptr = input + b * stride_batch + i * stride_dim1;

    // ------------------------------------------------------------------
    //  Step 1 – each thread loads a subset of the row and computes a local max
    // ------------------------------------------------------------------
    float thread_max = -FLT_MAX;
    for (int idx = threadIdx.x; idx < dim2; idx += TPB) {
        float val = row_ptr[idx];
        if (val > thread_max) thread_max = val;
    }

    // ------------------------------------------------------------------
    //  Step 2 – write the per‑thread maxima to shared memory
    // ------------------------------------------------------------------
    __shared__ float sdata[TPB];
    sdata[threadIdx.x] = thread_max;
    __syncthreads();

    // ------------------------------------------------------------------
    //  Step 3 – tree‑reduction inside shared memory (only the first warp
    //           participates after the first barrier)
    // ------------------------------------------------------------------
    #pragma unroll
    for (int stride = TPB/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float other = sdata[threadIdx.x + stride];
            if (other > sdata[threadIdx.x]) sdata[threadIdx.x] = other;
        }
        __syncthreads();
    }

    // ------------------------------------------------------------------
    //  Step 4 – write the final max for this row
    // ------------------------------------------------------------------
    if (threadIdx.x == 0) {
        // output layout: (batch, dim1) -> contiguous
        output[b * gridDim.y + i] = sdata[0];
    }
}

// ------------------------------------------------------------------
//  Dispatcher that chooses a thread‑block size that fits the problem.
// ------------------------------------------------------------------
void max_reduce_last_launcher(
    torch::Tensor input,
    torch::Tensor output,
    const int dim)               // only dim == 2 is supported
{
    // sanity checks
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "input must be 3‑D");
    TORCH_CHECK(dim == 2, "this kernel only implements reduction over the last dimension (dim=2)");

    const int batch = input.size(0);
    const int dim1  = input.size(1);
    const int dim2  = input.size(2);

    const int stride_batch = input.stride(0);
    const int stride_dim1  = input.stride(1);

    // Choose a reasonable TPB (threads per block). 256 works for any dim2 ≤ 4096.
    constexpr int TPB = 256;

    dim3 blocks(batch, dim1);
    dim3 threads(TPB);

    // Launch kernel
    max_reduce_last_dim<TPB><<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        dim2,
        stride_batch,
        stride_dim1
    );

    // Propagate possible launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}
"""

# ----------------------------------------------------------------------
#  C++ wrapper (pybind11) – registers the function as `max_reduce_last`
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the CUDA implementation
void max_reduce_last_launcher(torch::Tensor input,
                              torch::Tensor output,
                              const int dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduce_last", &max_reduce_last_launcher,
          "Max‑reduction over the last dimension using shared memory (CUDA)");
}
"""

# ----------------------------------------------------------------------
#  Build the extension (compile with -O3 and fast math)
# ----------------------------------------------------------------------
max_reduce_ext = load_inline(
    name='max_reduce_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False
)

# ----------------------------------------------------------------------
#  Public API – replaces the original `functional_model`
# ----------------------------------------------------------------------
def functional_model(x: torch.Tensor, *, dim: int) -> torch.Tensor:
    """
    Computes torch.max(x, dim=dim)[0] using a custom CUDA kernel that
    leverages shared memory.  Only `dim == 2` (last axis) is supported.
    """
    if dim != 2:
        raise ValueError("Only reduction over the last dimension (dim=2) is supported "
                         "by the custom kernel.")
    # Allocate output tensor: shape is x without the reduced dimension
    out_shape = list(x.shape)
    del out_shape[dim]
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    # Call the kernel
    max_reduce_ext.max_reduce_last(x, out, dim)

    return out

# ----------------------------------------------------------------------
#  Simple sanity test (will be run during evaluation)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    batch_size = 128
    dim1 = 4096
    dim2 = 4095

    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    # Reference result using PyTorch for verification
    ref = torch.max(x, dim=2)[0]

    # Result from our kernel
    out = functional_model(x, dim=2)

    # Verify numerical correctness (tolerance accounts for fp32 rounding)
    torch.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
    print("✅ Correctness check passed.")
