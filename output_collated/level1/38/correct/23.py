# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_030939/code_8.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Warp-level reduction helper using __shfl_down_sync
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel 1: per-row reduction using warp shuffles and minimal shared memory
__global__ void reduce_sum_abs_kernel(const float * __restrict__ x,
                                      float * __restrict__ sum_abs,
                                      const int batch_size,
                                      const int dim)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = (blockDim.x + 31) / 32;

    // Each thread accumulates partial sum of |x[row, col]|
    float sum = 0.0f;
    for (int col = tid; col < dim; col += stride) {
        float v = x[row * dim + col];
        sum += fabsf(v);
    }

    // Warp-level reduction (no synchronization needed within warp)
    sum = warp_reduce_sum(sum);

    // Store warp results in shared memory
    __shared__ float warp_sums[32];
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction across warps (only first warp does work)
    if (warp_id == 0) {
        float final_sum = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        final_sum = warp_reduce_sum(final_sum);
        if (lane_id == 0) {
            sum_abs[row] = final_sum;
        }
    }
}

// Kernel 2: element-wise scaling
__global__ void scale_kernel(const float * __restrict__ x,
                             const float * __restrict__ sum_abs,
                             float * __restrict__ output,
                             const int batch_size,
                             const int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;
    if (idx >= total) return;

    const int row = idx / dim;
    float sum = sum_abs[row];
    float val = x[idx];

    // Normalize: x * dim / sum_abs
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
    const int blocks     = batch_size;

    // Reduction kernel
    reduce_sum_abs_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), sum_abs.data_ptr<float>(),
        batch_size, dim);

    // Scaling kernel (no cudaDeviceSynchronize needed - implicit ordering on same stream)
    const int total = batch_size * dim;
    const int blocks_s = (total + threads - 1) / threads;
    scale_kernel<<<blocks_s, threads>>>(
        x.data_ptr<float>(), sum_abs.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, dim);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_normalize(torch::Tensor x,
                     torch::Tensor sum_abs,
                     torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize,
          "Fused abs-reduction + normalization with warp-level optimizations");
}
"""

fused_ext = load_inline(
    name='fused_norm_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize each row of x by the mean of its absolute values.
    Equivalent to: x / torch.mean(torch.abs(x), dim=1, keepdim=True)
    """
    if not x.is_cuda:
        x = x.cuda()

    batch_size = x.size(0)
    dim        = x.size(1)

    sum_abs = torch.empty(batch_size, dtype=x.dtype, device=x.device)
    output  = torch.empty_like(x)

    fused_ext.fused_normalize(x, sum_abs, output)

    return output


def get_init_inputs():
    return []

def get_inputs():
    batch_size = 32768
    dim = 65535
    x = torch.rand(batch_size, dim, dtype=torch.float32)
    return [x]
