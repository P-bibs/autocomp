# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_030939/code_27.py
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

# The kernel optimized for coalesced global memory access and efficient reductions.
cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_normalize_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const int N,
    const int D
) {
    int bid = blockIdx.x;
    if (bid >= N) return;

    // Phase 1: Compute L1 Norm
    // Each thread gathers a partial sum across the dimension D
    float local_sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        local_sum += fabsf(x[bid * D + d]);
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    // Shared memory to store warp-level reductions
    extern __shared__ float sdata[];
    if (threadIdx.x % 32 == 0) {
        sdata[threadIdx.x / 32] = local_sum;
    }
    __syncthreads();

    // Final reduction of warp sums
    if (threadIdx.x < 32) {
        float warp_sum = (threadIdx.x < (blockDim.x / 32)) ? sdata[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
        }
        if (threadIdx.x == 0) {
            sdata[0] = (float)D / warp_sum;
        }
    }
    __syncthreads();

    float mean_inv = sdata[0];

    // Phase 2: Compute Result (Coalesced write)
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        out[bid * D + d] = x[bid * D + d] * mean_inv;
    }
}

void launch_fused_normalize(const at::Tensor& x, at::Tensor& out) {
    const int N = x.size(0);
    const int D = x.size(1);
    const int threads = 256; // Optimized for occupancy on 2080Ti
    const int blocks = N;
    size_t smem = (threads / 32) * sizeof(float);
    
    fused_normalize_kernel<<<blocks, threads, smem>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), N, D
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

# Compile in-place
fused_module = load_inline(
    name='fused_normalize_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Optimized normalization:
    1. Ensures input is contiguous to maximize cache line utilization.
    2. Uses warpsync primitives for fast parallel reduction.
    3. Balanced thread count (256) for optimal occupancy on Turing architecture.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    fused_module.forward(x, out)
    return out
