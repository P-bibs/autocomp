# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_024918/code_27.py
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

# Optimized CUDA Kernel:
# - Uses warp-level primitives (__shfl_down_sync) to eliminate bank conflicts and synchronization overhead.
# - Reduces shared memory footprint to just one float per warp.
# - Uses 256 threads per block to balance occupancy and arithmetic throughput.
cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

__global__ void fused_normalize_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const int N,
    const int D
) {
    extern __shared__ float sdata[]; // Size will be (threads / 32)
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (bid >= N) return;

    // Phase 1: Local summation
    float local_sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        local_sum += fabsf(x[bid * D + d]);
    }

    // Warp-level reduction
    local_sum = warp_reduce_sum(local_sum);

    // Save warp results to shared memory
    if (lane_id == 0) {
        sdata[warp_id] = local_sum;
    }
    __syncthreads();

    // Final reduction of warp results
    float total_sum = 0.0f;
    if (tid < (blockDim.x / 32)) {
        total_sum = sdata[tid];
    }
    total_sum = warp_reduce_sum(total_sum);
    
    // Broadcast total_sum
    float mean_inv = (float)D / (__shfl_sync(0xffffffff, total_sum, 0));

    // Phase 2: Compute
    for (int d = tid; d < D; d += blockDim.x) {
        out[bid * D + d] = x[bid * D + d] * mean_inv;
    }
}

void launch_fused_normalize(const at::Tensor& x, at::Tensor& out) {
    const int N = x.size(0);
    const int D = x.size(1);
    const int threads = 256;
    const int warps = threads / 32;
    
    fused_normalize_forward_kernel<<<N, threads, warps * sizeof(float)>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        N, 
        D
    );
}
'''

cpp_source = r'''
#include <torch/extension.h>
void launch_fused_normalize(const at::Tensor& x, at::Tensor& out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launch_fused_normalize, "Fused Normalize Forward");
}
'''

# Compile JIT
fused_module = load_inline(
    name='fused_normalize',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Optimized functional_model using warp-level primitives.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Ensure input is float32 as per kernel expectations
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
        
    out = torch.empty_like(x)
    fused_module.forward(x, out)
    return out

def get_init_inputs():
    return []

def get_inputs():
    # Example dimensions matching the target requirement
    batch_size = 32768
    dim = 65535
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]
