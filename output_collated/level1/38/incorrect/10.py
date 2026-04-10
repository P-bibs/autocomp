# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_022930/code_18.py
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

# --- Optimized CUDA Kernel ---
# Uses warp-shuffle reduction for reduced latency and memory synchronization.
cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void fused_normalize_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const int N,
    const int D
) {
    extern __shared__ float sdata[];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    if (bid >= N) return;

    // Phase 1: Local accumulation per thread
    float local_sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        local_sum += fabsf(x[bid * D + d]);
    }

    // Phase 2: Warp-level reduction
    local_sum = warpReduceSum(local_sum);

    // Phase 3: Block-level reduction
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    if (lane_id == 0) {
        sdata[warp_id] = local_sum;
    }
    __syncthreads();

    // Reduce the partial sums across warps (at most 512/32 = 16 warps)
    float final_sum = (tid < (blockDim.x / 32)) ? sdata[tid] : 0.0f;
    if (warp_id == 0) {
        final_sum = warpReduceSum(final_sum);
    }

    // Broadcast the result to all threads
    float mean_inv = (final_sum > 0.0f) ? (float)D / final_sum : 0.0f;

    // Phase 4: Normalize
    for (int d = tid; d < D; d += blockDim.x) {
        out[bid * D + d] = x[bid * D + d] * mean_inv;
    }
}

void launch_fused_normalize(const at::Tensor& x, at::Tensor& out) {
    const int N = x.size(0);
    const int D = x.size(1);
    const int threads = 256; // Optimized occupancy
    const int smem = (threads / 32) * sizeof(float);
    
    fused_normalize_forward_kernel<<<N, threads, smem>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        N, 
        D
    );
}
'''

# --- C++ Binding ---
cpp_source = r'''
#include <torch/extension.h>
void launch_fused_normalize(const at::Tensor& x, at::Tensor& out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launch_fused_normalize, "Fused Normalize Forward");
}
'''

# Compile
fused_module = load_inline(
    name='fused_normalize',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Optimized functional_model using JIT compiled CUDA kernel.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    fused_module.forward(x, out)
    return out

# --- Compatibility requirements ---
batch_size = 32768
dim = 65535

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]
