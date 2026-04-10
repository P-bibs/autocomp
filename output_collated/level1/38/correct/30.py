# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_032736/code_0.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- CUDA Kernel Code ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

__device__ __forceinline__ float warpReduceSum(float val) {
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

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const float* x_vec = input + batch_idx * dim;
    float* out_vec = output + batch_idx * dim;

    // Phase 1: Compute sum of absolute values with improved memory coalescing
    float local_sum = 0.0f;
    
    // Each thread processes multiple elements in a coalesced manner
    for (int i = tid; i < dim; i += stride) {
        local_sum += fabsf(x_vec[i]);
    }

    // Warp-level reduction
    local_sum = warpReduceSum(local_sum);

    // Final reduction using shared memory within the block
    __shared__ float sdata[32]; // Assumes <= 1024 threads, so 32 warps max
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (lane_id == 0) {
        sdata[warp_id] = local_sum;
    }
    __syncthreads();

    // Load warp sums into lane 0 of first warp and reduce
    if (warp_id == 0) {
        float warp_sum = (lane_id < (blockDim.x + 31) / 32) ? sdata[lane_id] : 0.0f;
        warp_sum = warpReduceSum(warp_sum);
        if (lane_id == 0) {
            sdata[0] = warp_sum;
        }
    }
    __syncthreads();

    // Normalize with improved memory coalescing
    const float mean_abs = sdata[0] / static_cast<float>(dim);
    const float inv_mean = 1.0f / (mean_abs + 1e-8f);

    // Process memory in coalesced manner - each thread processes stride-separated elements
    for (int i = tid; i < dim; i += stride) {
        out_vec[i] = x_vec[i] * inv_mean;
    }
}

void fused_normalize_forward(
    const torch::Tensor input,
    torch::Tensor output,
    const int batch_size,
    const int dim
) {
    const dim3 threads(512);
    const dim3 blocks(batch_size);

    fused_normalize_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_normalize_forward(
    const torch::Tensor input,
    torch::Tensor output,
    const int batch_size,
    const int dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize_forward, "Fused normalize forward with optimized memory coalescing");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_normalize_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
    with_cuda=True
)

def functional_model(x):
    # Ensure input is contiguous and on GPU
    if not x.is_cuda:
        x = x.cuda()
    x = x.contiguous()
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Launch fused kernel
    fused_ext.fused_normalize(x, output, x.size(0), x.size(1))
    
    return output

batch_size = 32768
dim = 65535

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]
