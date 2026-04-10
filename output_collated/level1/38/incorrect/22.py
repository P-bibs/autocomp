# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_032736/code_22.py
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
# CUDA Source: Optimized using Warp-Level Primitives.
# The original code suffered from launch overhead by calling two separate 
# kernels. This implementation fuses the reduction and scaling into a single 
# kernel pass per row, reducing global memory round-trips.
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fused_normalize_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
    const int batch_size,
    const int dim) 
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    float thread_sum = 0.0f;
    // 1. Partial reduction per thread
    for (int col = tid; col < dim; col += stride) {
        thread_sum += fabsf(x[row * dim + col]);
    }

    // 2. Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // 3. Block-level reduction using shared memory
    __shared__ float sdata[32]; // Max 1024 threads / 32 threads per warp = 32
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    if (lane == 0) sdata[wid] = thread_sum;
    __syncthreads();

    // The first warp reduces the partial sums from all warps
    float block_sum = 0.0f;
    if (wid == 0) {
        block_sum = (lane < (blockDim.x / warpSize)) ? sdata[lane] : 0.0f;
        for (int offset = (blockDim.x / warpSize) / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
    }

    // 4. Final broadcast of the total sum for this row
    float row_sum = __shfl_sync(0xffffffff, block_sum, 0);
    float inv_sum = (float)dim / row_sum;

    // 5. Scaling: re-read x and compute normalized output
    for (int col = tid; col < dim; col += stride) {
        output[row * dim + col] = x[row * dim + col] * inv_sum;
    }
}

void fused_normalize(torch::Tensor x, torch::Tensor output) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);
    
    // 512 threads is ideal for 2080Ti for occupancy vs shared memory usage
    const int threads = 512;
    fused_normalize_kernel<<<batch_size, threads>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_normalize(torch::Tensor x, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize, "Optimized fused normalization kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_norm_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized normalization: one-pass kernel that calculates row-wise L1 
    normalizer and scales the data, minimizing global memory round-trips.
    """
    if not x.is_cuda:
        x = x.cuda()
    
    output = torch.empty_like(x)
    fused_ext.fused_normalize(x, output)
    return output

def get_init_inputs():
    return []

def get_inputs():
    # Shape matching the problem definition
    return [torch.rand(32768, 65535, dtype=torch.float32)]
