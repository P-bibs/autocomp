# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_022930/code_0.py
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
# CUDA source – two fused kernels:
#   1. reduce_sum_abs_kernel – computes per‑row sum of |x| using warp shuffle
#   2. scale_kernel           – normalizes each element: x * dim / sum_abs
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel 1: per‑row reduction – one block per row
__global__ void reduce_sum_abs_kernel(const float *x,
                                      float *sum_abs,
                                      const int batch_size,
                                      const int dim)
{
    const int row = blockIdx.x;                 // one block per row
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // Each thread accumulates a partial sum of |x[row, col]|
    float sum = 0.0f;
    for (int col = tid; col < dim; col += stride) {
        float v = x[row * dim + col];
        sum += fabsf(v);
    }

    // Store partial sum in shared memory
    __shared__ float sdata[256];
    sdata[tid] = sum;
    __syncthreads();

    // Parallel reduction within the block until 32 elements remain
    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Final warp shuffle reduction
    if (tid < 32) {
        float warp_sum = sdata[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
        }

        if (tid == 0) {
            sum_abs[row] = warp_sum;
        }
    }
}

// Kernel 2: element‑wise scaling – one thread per element
__global__ void scale_kernel(const float *x,
                             const float *sum_abs,
                             float *output,
                             const int batch_size,
                             const int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;
    if (idx >= total) return;

    const int row = idx / dim;
    // const int col = idx % dim; // not needed explicitly

    float sum = sum_abs[row];
    float val = x[idx];

    // Normalize: x * dim / sum_abs
    // Division by zero yields +/-inf as in the original PyTorch code
    float out = val * static_cast<float>(dim) / sum;
    output[idx] = out;
}

// Host function that launches both kernels
void fused_normalize(torch::Tensor x,
                     torch::Tensor sum_abs,
                     torch::Tensor output)
{
    const int batch_size = x.size(0);
    const int dim        = x.size(1);
    const int threads    = 256;
    const int blocks     = batch_size;                // one block per row

    // ----- reduction -----
    reduce_sum_abs_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), sum_abs.data_ptr<float>(),
        batch_size, dim);
    cudaDeviceSynchronize();

    // ----- scaling -----
    const int total = batch_size * dim;
    const int blocks_s = (total + threads - 1) / threads;
    scale_kernel<<<blocks_s, threads>>>(
        x.data_ptr<float>(), sum_abs.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, dim);
    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
# C++ binding – exposes fused_normalize to Python
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_normalize(torch::Tensor x,
                     torch::Tensor sum_abs,
                     torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize,
          "Fused abs‑reduction + normalization kernel with warp shuffle");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_norm',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# functional_model – the only function imported during evaluation
# ----------------------------------------------------------------------
def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize each row of x by the mean of its absolute values.
    Equivalent to: x / torch.mean(torch.abs(x), dim=1, keepdim=True)
    """
    # Ensure the input lives on the GPU
    if not x.is_cuda:
        x = x.cuda()

    batch_size = x.size(0)
    dim        = x.size(1)

    # Allocate temporary buffer for per‑row sums and output
    sum_abs = torch.empty(batch_size, dtype=x.dtype, device=x.device)
    output  = torch.empty_like(x)

    # Launch fused kernels
    fused_ext.fused_normalize(x, sum_abs, output)

    return output


# ----------------------------------------------------------------------
# Helper functions required by the harness
# ----------------------------------------------------------------------
def get_init_inputs():
    return []

def get_inputs():
    batch_size = 32768
    dim = 65535
    # Random input matching the original benchmark shape/dtype
    x = torch.rand(batch_size, dim, dtype=torch.float32)
    return [x]
