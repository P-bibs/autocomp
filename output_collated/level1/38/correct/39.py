# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_032736/code_23.py
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
    // Grid-stride loop: each block processes multiple rows
    for (int batch_idx = blockIdx.x; batch_idx < batch_size; batch_idx += gridDim.x) {
        const float* x_vec = input + (size_t)batch_idx * dim;
        float* out_vec = output + (size_t)batch_idx * dim;

        float local_sum = 0.0f;
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            local_sum += fabsf(x_vec[i]);
        }

        local_sum = warpReduceSum(local_sum);
        
        static __shared__ float sdata[32];
        if (threadIdx.x % 32 == 0) sdata[threadIdx.x / 32] = local_sum;
        __syncthreads();
        
        if (threadIdx.x < 32) {
            float val = (threadIdx.x < (blockDim.x / 32)) ? sdata[threadIdx.x] : 0.0f;
            val = warpReduceSum(val);
            if (threadIdx.x == 0) sdata[0] = val;
        }
        __syncthreads();

        float inv_mean = (float)dim / (sdata[0] + 1e-8f);
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            out_vec[i] = x_vec[i] * inv_mean;
        }
    }
}

void fused_normalize_forward(const torch::Tensor input, torch::Tensor output) {
    const int batch_size = input.size(0);
    const int dim = input.size(1);
    const int threads = 256;
    // Cap blocks to prevent over-subscription (68 SMs on 2080Ti)
    const int blocks = std::min(batch_size, 1024);
    
    fused_normalize_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );
}
"""

# --- C++ Logic ---
cpp_source = r"""
#include <torch/extension.h>
void fused_normalize_forward(const torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize_forward, "Fused normalize");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_normalize_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    # Ensure input is on device and contiguous
    x = x.contiguous().cuda()
    output = torch.empty_like(x)
    
    # Launch fused kernel
    fused_ext.fused_normalize(x, output)
    
    return output
