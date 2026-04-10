# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_032736/code_4.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_normalize_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int dim
) {
    // Use shared memory for reduction
    extern __shared__ float sdata[];
    
    // Grid-stride loop: each thread block processes multiple rows
    for (int batch_idx = blockIdx.x; batch_idx < batch_size; batch_idx += gridDim.x) {
        const float* x_row = input + (size_t)batch_idx * dim;
        float* out_row = output + (size_t)batch_idx * dim;
        
        // Phase 1: Compute sum of absolute values
        float local_sum = 0.0f;
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            local_sum += fabsf(x_row[i]);
        }
        
        // Warp-level reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }
        
        // Store warp results in shared memory
        const int warp_id = threadIdx.x / 32;
        const int lane_id = threadIdx.x % 32;
        
        if (lane_id == 0) {
            sdata[warp_id] = local_sum;
        }
        __syncthreads();
        
        // Final reduction of warp sums (only first warp participates)
        if (warp_id == 0) {
            float warp_sum = (lane_id < (blockDim.x + 31) / 32) ? sdata[lane_id] : 0.0f;
            for (int offset = 16; offset > 0; offset /= 2) {
                warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
            }
            
            // Broadcast result to all threads in the block
            if (lane_id == 0) {
                sdata[0] = warp_sum;
            }
        }
        __syncthreads();
        
        // Normalize with coalesced memory access
        const float mean_abs = sdata[0] / static_cast<float>(dim);
        const float inv_mean = 1.0f / (mean_abs + 1e-8f);
        
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            out_row[i] = x_row[i] * inv_mean;
        }
    }
}

void fused_normalize_forward(
    const torch::Tensor input,
    torch::Tensor output,
    const int batch_size,
    const int dim
) {
    // Use fewer blocks to reduce kernel launch overhead
    const int threads_per_block = 512;
    const int blocks_per_grid = min(65535, (batch_size + 15) / 16); // Cap at reasonable block count
    
    // Shared memory for reduction: 32 warps max * sizeof(float)
    fused_normalize_kernel<<<blocks_per_grid, threads_per_block, 32 * sizeof(float)>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_normalize_forward(
    const torch::Tensor input,
    torch::Tensor output,
    const int batch_size,
    const int dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize_forward, "Fused normalize with grid-stride optimization");
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
    # Ensure input is on GPU and contiguous
    if not x.is_cuda:
        x = x.cuda()
    x = x.contiguous()
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Launch fused kernel
    fused_ext.fused_normalize(x, output, x.size(0), x.size(1))
    
    return output
