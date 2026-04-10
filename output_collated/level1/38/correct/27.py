# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_030939/code_18.py
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
# We use one block per row for high D, ensuring we maximize occupancy.
# 1024 threads per block allow us to saturate the memory bandwidth.
cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void fused_normalize_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const int N,
    const int D
) {
    extern __shared__ float sdata[];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    if (bid >= N) return;

    const float* row_x = x + bid * D;
    float* row_out = out + bid * D;

    float local_sum = 0.0f;
    // Coalesced loading
    for (int d = tid; d < D; d += blockDim.x) {
        local_sum += fabsf(row_x[d]);
    }

    // Reduce sum across threads in the block
    local_sum = warp_reduce_sum(local_sum);
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) sdata[warp_id] = local_sum;
    __syncthreads();

    // Final block reduction
    if (warp_id == 0) {
        float sum = (tid < (blockDim.x / WARP_SIZE)) ? sdata[tid] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (tid == 0) sdata[0] = sum;
    }
    __syncthreads();

    float inv_sum = (float)D / sdata[0];

    // Final normalization pass
    for (int d = tid; d < D; d += blockDim.x) {
        row_out[d] = row_x[d] * inv_sum;
    }
}

void launch_fused_normalize(const at::Tensor& x, at::Tensor& out) {
    const int N = x.size(0);
    const int D = x.size(1);
    const int threads = 512;
    const int blocks = N;
    const size_t shared_mem = (threads / WARP_SIZE) * sizeof(float);
    
    fused_normalize_kernel<<<blocks, threads, shared_mem>>>(
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

fused_module = load_inline(
    name='fused_normalize',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    fused_module.forward(x, out)
    return out

batch_size = 32768
dim = 65535

def get_init_inputs(): return []
def get_inputs(): return [torch.rand(batch_size, dim, device='cuda')]
