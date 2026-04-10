# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_022930/code_1.py
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

#define WARP_SIZE 32
#define MAX_SHARED_MEMORY 49152  // 48KB for RTX 2080Ti

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

    // Calculate how much shared memory we can use per thread block
    // We need to store both the data and reduction array
    const int max_shared_elements = (MAX_SHARED_MEMORY / sizeof(float)) / 2;
    
    // Dynamically adjust tile size based on available shared memory
    const int tile_size = min(dim, max_shared_elements);
    
    extern __shared__ float shared_mem[];
    float* shared_data = shared_mem;
    float* shared_sums = shared_mem + tile_size;

    float total_sum = 0.0f;
    
    // Process input in tiles
    for (int tile_start = 0; tile_start < dim; tile_start += tile_size) {
        const int current_tile_size = min(tile_size, dim - tile_start);
        
        // Phase 1: Load tile data into shared memory and compute partial sums
        float local_sum = 0.0f;
        for (int i = tid; i < current_tile_size; i += stride) {
            int global_idx = tile_start + i;
            if (global_idx < dim) {
                float val = x_vec[global_idx];
                shared_data[i] = val;
                local_sum += fabsf(val);
            }
        }
        
        // Warp-level reduction of local sums
        local_sum = warpReduceSum(local_sum);
        
        // Store warp sums in shared memory
        const int warp_id = tid / WARP_SIZE;
        const int lane_id = tid % WARP_SIZE;
        
        if (lane_id == 0) {
            shared_sums[warp_id] = local_sum;
        }
        __syncthreads();
        
        // Final reduction across warps
        if (warp_id == 0) {
            float warp_sum = (lane_id < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? shared_sums[lane_id] : 0.0f;
            warp_sum = warpReduceSum(warp_sum);
            if (lane_id == 0) {
                total_sum += warp_sum;
            }
        }
        __syncthreads();
        
        // Normalize tile data using cached values
        const float mean_abs = total_sum / static_cast<float>(dim);
        const float inv_mean = 1.0f / (mean_abs + 1e-8f);
        
        for (int i = tid; i < current_tile_size; i += stride) {
            int global_idx = tile_start + i;
            if (global_idx < dim) {
                out_vec[global_idx] = shared_data[i] * inv_mean;
            }
        }
        __syncthreads();
    }
}

void fused_normalize_forward(
    const torch::Tensor input,
    torch::Tensor output,
    const int batch_size,
    const int dim
) {
    const dim3 threads(256);
    const dim3 blocks(batch_size);

    // Calculate shared memory size needed - data + reduction array
    const int tile_size = min(dim, 16384); // Conservative estimate
    const size_t shared_mem_size = tile_size * 2 * sizeof(float);

    fused_normalize_kernel<<<blocks, threads, shared_mem_size>>>(
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
    m.def("fused_normalize", &fused_normalize_forward, "Fused normalize forward");
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
