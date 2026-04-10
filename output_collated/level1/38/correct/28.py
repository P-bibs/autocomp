# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_030939/code_24.py
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
# Optimized CUDA implementation
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Warp-level reduction helper using __shfl_down_sync
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized Kernel: Per-row absolute sum reduction
// Each block handles one row (batch item)
__global__ void reduce_sum_abs_kernel(const float * __restrict__ x,
                                      float * __restrict__ sum_abs,
                                      const int dim)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    float sum = 0.0f;
    // Coalesced access: threads within a warp read contiguous indices
    for (int col = tid; col < dim; col += blockDim.x) {
        sum += fabsf(x[row * dim + col]);
    }

    // Warp-level reduction
    sum = warp_reduce_sum(sum);

    // Shared memory for reduction across warps
    __shared__ float warp_sums[32];
    int lane_id = tid % 32;
    int warp_id = tid / 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        float final_val = (tid < (blockDim.x / 32)) ? warp_sums[tid] : 0.0f;
        final_val = warp_reduce_sum(final_val);
        if (tid == 0) {
            sum_abs[row] = final_val;
        }
    }
}

// Optimized Kernel: Element-wise scaling
// Grid size is maximized for global occupancy
__global__ void scale_kernel(const float * __restrict__ x,
                             const float * __restrict__ sum_abs,
                             float * __restrict__ output,
                             const int total_elements,
                             const int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    const int row = idx / dim;
    // Normalize: x * dim / sum_abs
    // Optimization: Multiply by (dim / sum)
    output[idx] = x[idx] * (static_cast<float>(dim) / sum_abs[row]);
}

void fused_normalize_impl(torch::Tensor x, torch::Tensor sum_abs, torch::Tensor output) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);
    
    // Launch Reduction
    // 256 threads is generally ideal for 2080Ti/Ampere architectures
    reduce_sum_abs_kernel<<<batch_size, 256>>>(x.data_ptr<float>(), sum_abs.data_ptr<float>(), dim);
    
    // Launch Scaling
    const int total = batch_size * dim;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    scale_kernel<<<blocks, threads>>>(x.data_ptr<float>(), sum_abs.data_ptr<float>(), output.data_ptr<float>(), total, dim);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_normalize_impl(torch::Tensor x, torch::Tensor sum_abs, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize_impl, "Fused abs-reduction and scaling");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_norm_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        x = x.cuda()

    batch_size, dim = x.size()
    sum_abs = torch.empty(batch_size, dtype=x.dtype, device=x.device)
    output = torch.empty_like(x)

    # Launch kernel
    fused_ext.fused_normalize(x, sum_abs, output)
    
    return output

def get_init_inputs():
    return []

def get_inputs():
    return [torch.rand(32768, 65535, dtype=torch.float32)]
