# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_022930/code_17.py
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

# --- CUDA Kernel Code ---
# The primary optimization here is a block-wide reduction that ensures 
# minimal global memory traffic and avoids the shared memory size limit.
# Since dim=65535, we process the row using a grid-stride approach 
# with a single-pass sum calculation.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void fused_normalize_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int dim
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float* x_vec = input + batch_idx * dim;
    float* out_vec = output + batch_idx * dim;

    float local_sum = 0.0f;
    // Single pass: Read data once from global memory
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local_sum += fabsf(x_vec[i]);
    }

    // Block-wide reduction
    static __shared__ float sdata[32];
    local_sum = warpReduceSum(local_sum);
    
    if ((threadIdx.x & 31) == 0) sdata[threadIdx.x >> 5] = local_sum;
    __syncthreads();

    local_sum = (threadIdx.x < (blockDim.x >> 5)) ? sdata[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) local_sum = warpReduceSum(local_sum);

    const float mean_abs = sdata[0] / static_cast<float>(dim);
    const float inv_mean = 1.0f / (mean_abs + 1e-8f);

    // Apply normalization
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out_vec[i] = x_vec[i] * inv_mean;
    }
}

void fused_normalize_forward(const torch::Tensor input, torch::Tensor output) {
    const int batch_size = input.size(0);
    const int dim = input.size(1);
    const int threads = 256;
    fused_normalize_kernel<<<batch_size, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_normalize_forward(const torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize_forward, "Fused normalize");
}
"""

fused_ext = load_inline(
    name='fused_normalize_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    output = torch.empty_like(x)
    fused_ext.fused_normalize(x.contiguous(), output)
    return output
