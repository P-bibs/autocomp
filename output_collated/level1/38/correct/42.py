# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_032736/code_30.py
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
# CUDA source – optimized fused kernels
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

// Warp‑level reduction using shuffle intrinsics
static __device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel 1: Per‑row absolute sum reduction
__global__ void reduce_sum_abs_kernel(const float *__restrict__ x,
                                      float *__restrict__ sum_abs,
                                      const int dim)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    float sum = 0.0f;
    // Coalesced loading: processes elements in a stride to maximize throughput
    for (int col = tid; col < dim; col += blockDim.x) {
        sum += fabsf(x[row * dim + col]);
    }

    // Warp-level reduction
    float warpSum = warpReduceSum(sum);

    // Shared memory for warp results (8 warps per 256 threads)
    __shared__ float sdata[8];
    if (tid % WARP_SIZE == 0) {
        sdata[tid / WARP_SIZE] = warpSum;
    }
    __syncthreads();

    // Final reduction of the 8 warp partial sums
    if (tid < 8) {
        float val = sdata[tid];
        // Short reduction for remaining 8 values
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0x000000FF, val, offset);
        }
        if (tid == 0) {
            sum_abs[row] = val;
        }
    }
}

// Kernel 2: Element‑wise scaling
__global__ void scale_kernel(const float *__restrict__ x,
                             const float *__restrict__ sum_abs,
                             float *__restrict__ output,
                             const int batch_size,
                             const int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * dim) return;

    const int row = idx / dim;
    // Normalize: x * dim / sum_abs
    output[idx] = x[idx] * (static_cast<float>(dim) / sum_abs[row]);
}

void fused_normalize(torch::Tensor x, torch::Tensor sum_abs, torch::Tensor output) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);
    const int threads = 256;
    
    reduce_sum_abs_kernel<<<batch_size, threads>>>(x.data_ptr<float>(), sum_abs.data_ptr<float>(), dim);
    
    const int total = batch_size * dim;
    const int blocks_s = (total + threads - 1) / threads;
    scale_kernel<<<blocks_s, threads>>>(x.data_ptr<float>(), sum_abs.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_normalize(torch::Tensor x, torch::Tensor sum_abs, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize, "Fused abs‑reduction + normalization kernel");
}
"""

fused_ext = load_inline(
    name='fused_norm_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda: x = x.cuda()
    batch_size, dim = x.size()
    sum_abs = torch.empty(batch_size, dtype=x.dtype, device=x.device)
    output = torch.empty_like(x)
    fused_ext.fused_normalize(x, sum_abs, output)
    return output

def get_init_inputs(): return []
def get_inputs():
    return [torch.rand(32768, 65535, dtype=torch.float32)]
