# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_020747/code_6.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs L1 normalization.
    """

    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(ModelNew, self).__init__()

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
# CUDA source – fused kernel that performs both the reduction of |x|
# and the element‑wise scaling in a single launch.
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

constexpr int BLOCK_SIZE = 256;          // threads per block (one block per row)

__global__ void fused_normalize_kernel(const float* __restrict__ x,
                                       float* __restrict__ output,
                                       const int dim)
{
    const int row = blockIdx.x;          // one block per row
    const int tid = threadIdx.x;

    // Dynamic shared memory for the per‑block reduction
    extern __shared__ float sdata[];

    // ----------------------------------------------------------
    // Phase 1: compute partial sum of |x[row, col]|
    // ----------------------------------------------------------
    float sum = 0.0f;
    for (int col = tid; col < dim; col += blockDim.x) {
        // __ldg leverages the read‑only data cache
        float val = __ldg(&x[row * dim + col]);
        sum += fabsf(val);
    }
    sdata[tid] = sum;
    __syncthreads();

    // ----------------------------------------------------------
    // Parallel reduction within the block
    // ----------------------------------------------------------
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Row sum is now in sdata[0] – broadcast to all threads
    float row_sum = sdata[0];

    // ----------------------------------------------------------
    // Phase 2: scale each element using the row sum
    // ----------------------------------------------------------
    for (int col = tid; col < dim; col += blockDim.x) {
        int idx   = row * dim + col;
        float val = __ldg(&x[idx]);
        // Normalise: x * dim / sum_abs  (division by zero yields +/-inf as in the original)
        float out = val * static_cast<float>(dim) / row_sum;
        output[idx] = out;
    }
}

// Host function that launches the fused kernel
void fused_normalize(torch::Tensor x, torch::Tensor output)
{
    const int batch_size = x.size(0);
    const int dim        = x.size(1);

    const int grid   = batch_size;                // one block per row
    const int block  = BLOCK_SIZE;
    const int shared = block * sizeof(float);     // 256 floats

    fused_normalize_kernel<<<grid, block, shared>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        dim);

    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
# C++ binding – exposes fused_normalize to Python
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_normalize(torch::Tensor x, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize,
          "Fused abs‑reduction + normalisation kernel");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_norm',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# functional_model – the only function imported during evaluation
# ----------------------------------------------------------------------
def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Normalise each row of x by the sum of its absolute values.
    Equivalent to: x * dim / torch.sum(torch.abs(x), dim=1, keepdim=True)
    """
    # Move input to GPU if not already there
    if not x.is_cuda:
        x = x.cuda()

    batch_size = x.size(0)
    dim        = x.size(1)

    # Allocate output tensor on the same device
    output = torch.empty_like(x)

    # Launch the fused CUDA kernel
    fused_ext.fused_normalize(x, output)

    return output


# ----------------------------------------------------------------------
# Helper functions required by the harness
# ----------------------------------------------------------------------
def get_init_inputs():
    """No persistent state is required."""
    return []


def get_inputs():
    """Generate a random input matching the benchmark shape."""
    batch_size = 32768
    dim = 65535
    # Random float32 tensor on CPU; will be moved to GPU inside functional_model
    x = torch.rand(batch_size, dim, dtype=torch.float32)
    return [x]
