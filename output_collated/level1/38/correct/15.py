# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_022930/code_20.py
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
# Optimized CUDA Kernel Implementation
# The kernel uses warp-level primitives to reduce memory footprint and 
# latency. By performing the normalization within the same kernel, we 
# eliminate intermediate global memory writes and kernel launch overhead.
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Performs a reduction across the threads in a warp
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void fused_normalize_kernel(const float *__restrict__ x,
                                       float *__restrict__ output,
                                       const int dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    // 1. Calculate sum of absolute values
    float local_sum = 0.0f;
    for (int col = tid; col < dim; col += blockDim.x) {
        local_sum += fabsf(x[row * dim + col]);
    }
    
    // 2. Reduce within warps
    float warp_sum = warpReduceSum(local_sum);
    
    // 3. Reduce across warps using shared memory
    __shared__ float shared_sums[32];
    int lane = tid % 32;
    int wid = tid / 32;
    if (lane == 0) shared_sums[wid] = warp_sum;
    __syncthreads();
    
    // 4. Final aggregation of the partial reductions (if block > 1 warp)
    float row_sum = (tid < blockDim.x / 32) ? shared_sums[lane] : 0.0f;
    if (wid == 0) row_sum = warpReduceSum(row_sum);
    
    // Broadcast the final sum for this row
    __shared__ float final_sum;
    if (tid == 0) final_sum = row_sum;
    __syncthreads();
    
    // 5. Normalization - Write directly to output buffer
    const float scale = static_cast<float>(dim) / final_sum;
    for (int col = tid; col < dim; col += blockDim.x) {
        output[row * dim + col] = x[row * dim + col] * scale;
    }
}

void fused_normalize(torch::Tensor x, torch::Tensor output) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);
    // 256 threads is optimal for an occupancy of 8 on RTX 2080 Ti
    const int threads = 256;
    fused_normalize_kernel<<<batch_size, threads>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), dim);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_normalize(torch::Tensor x, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize, "Fused abs-reduction and normalization");
}
"""

# Compile the extension just-in-time
fused_ext = load_inline(
    name='fused_norm_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized normalization functional model.
    x: Input tensor of shape (batch, dim)
    Returns: Normalized tensor
    """
    if not x.is_cuda:
        x = x.cuda()
    
    output = torch.empty_like(x)
    fused_ext.fused_normalize(x, output)
    return output

def get_init_inputs():
    return []

def get_inputs():
    # Example dimensions matching target performance profile
    return [torch.rand(32768, 65535, dtype=torch.float32).cuda()]
