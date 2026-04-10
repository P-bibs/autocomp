# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_032736/code_6.py
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

// Warp-level reduction using shuffle operations
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized kernel fusing reduction and normalization
__global__ void fused_normalize_kernel(const float* __restrict__ x,
                                       float* __restrict__ output,
                                       const int batch_size,
                                       const int dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = (blockDim.x + 31) / 32;

    // Shared memory to store per-warp sums
    __shared__ float warp_sums[32]; // Max 32 warps per block

    // Each thread accumulates partial sum of absolute values
    float sum = 0.0f;
    for (int col = tid; col < dim; col += blockDim.x) {
        sum += fabsf(x[row * dim + col]);
    }

    // Reduce within warp using warp-level primitives
    sum = warp_reduce_sum(sum);

    // Write warp sum to shared memory
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // Single warp reduces the warp sums
    if (warp_id == 0) {
        sum = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        // Broadcast result to all threads in block
        if (lane_id == 0) {
            warp_sums[0] = sum;
        }
    }
    __syncthreads();

    // Get the final row sum
    const float row_sum = warp_sums[0];
    const float scale = static_cast<float>(dim) / row_sum;

    // Normalize elements
    for (int col = tid; col < dim; col += blockDim.x) {
        output[row * dim + col] = x[row * dim + col] * scale;
    }
}

void fused_normalize(torch::Tensor x, torch::Tensor output) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);
    // Use 256 threads for better occupancy on modern GPUs
    const int threads = 256;
    const int blocks = batch_size;

    fused_normalize_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_normalize(torch::Tensor x, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize, "Optimized fused normalization with warp-level primitives");
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
    Normalize each row of x by the mean of its absolute values.
    Equivalent to: x / torch.mean(torch.abs(x), dim=1, keepdim=True)
    """
    if not x.is_cuda:
        x = x.cuda()
    
    output = torch.empty_like(x)
    fused_ext.fused_normalize(x, output)
    return output

def get_init_inputs():
    return []

def get_inputs():
    batch_size = 32768
    dim = 65535
    x = torch.rand(batch_size, dim, dtype=torch.float32)
    return [x]
