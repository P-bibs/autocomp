# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_030939/code_28.py
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
# CUDA source – A single fused kernel that performs reduction and scaling.
# We utilize warp-level primitives (shuffles) to optimize the reduction 
# process, significantly reducing shared memory synchronization overheads
# and latency compared to global-memory-based approaches.
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void fused_normalize_kernel(const float *__restrict__ x,
                                       float *__restrict__ output,
                                       const int rows,
                                       const int cols)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Shared memory for warp-level partial sums
    __shared__ float shared[32];
    int lane = tid % 32;
    int wid  = tid / 32;

    // Phase 1: Compute per-row sum of |x| using warp reductions
    float thread_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        thread_sum += fabsf(x[row * cols + col]);
    }

    thread_sum = warpReduceSum(thread_sum);

    if (lane == 0) shared[wid] = thread_sum;
    __syncthreads();

    // Final sum reduction across warps
    float row_sum = (tid < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) row_sum = warpReduceSum(row_sum);
    
    // Broadcast total sum back to all threads
    const float total_sum = __shfl_sync(0xffffffff, row_sum, 0);

    // Phase 2: Scale and write result
    const float factor = static_cast<float>(cols) / (total_sum + 1e-20f);
    for (int col = tid; col < cols; col += blockDim.x) {
        output[row * cols + col] = x[row * cols + col] * factor;
    }
}

void fused_normalize(torch::Tensor x, torch::Tensor output) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);
    // Use 256 threads to maximize occupancy on RTX 2080 Ti
    const int threads = 256;
    
    fused_normalize_kernel<<<batch_size, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim);
}
"""

# ----------------------------------------------------------------------
# C++ binding
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_normalize(torch::Tensor x, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize, "Fused normalization kernel");
}
"""

# Compile the inline CUDA extension
fused_ext = load_inline(
    name='fused_norm_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized normalization: one-pass fused kernel using warp-shuffles.
    """
    if not x.is_cuda:
        x = x.cuda()

    output = torch.empty_like(x)
    fused_ext.fused_normalize(x, output)
    return output

def get_init_inputs():
    return []

def get_inputs():
    # Shape from original benchmark
    batch_size = 32768
    dim = 65535
    x = torch.rand(batch_size, dim, dtype=torch.float32)
    return [x]
