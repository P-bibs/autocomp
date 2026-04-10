# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_211408/code_10.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['negative_slope']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['negative_slope']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a LeakyReLU activation.
    """

    def __init__(self, negative_slope: float=0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope

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
    if 'negative_slope' in flat_state:
        state_kwargs['negative_slope'] = flat_state['negative_slope']
    else:
        state_kwargs['negative_slope'] = getattr(model, 'negative_slope')
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

# Optimized CUDA Kernel with shared memory tiling and float4 vectorization
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 1024
#define ELEMENTS_PER_THREAD 16
#define FLOAT4_PER_THREAD 4

__global__ void leaky_relu_vectorized_tiled_kernel(const float* __restrict__ input, 
                                                   float* __restrict__ output, 
                                                   float negative_slope, 
                                                   size_t n) {
    // Shared memory for tile
    __shared__ float shared_input[TILE_SIZE];
    
    // Thread and block indices
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x; // 256 threads per block
    
    // Each thread processes 16 floats (4 float4 vectors)
    // Each block processes 256 * 16 = 4096 floats
    size_t block_start = bid * TILE_SIZE;
    
    // Process data in tiles
    for (size_t tile_start = block_start; tile_start < n; tile_start += gridDim.x * TILE_SIZE) {
        // Load data into shared memory in a coalesced manner
        // Each thread loads 4 float4s (16 floats) 
        for (int i = 0; i < FLOAT4_PER_THREAD; i++) {
            int global_idx = tile_start + tid * FLOAT4_PER_THREAD + i;
            int shared_idx = tid * FLOAT4_PER_THREAD + i;
            
            if (global_idx < n && shared_idx < TILE_SIZE/4) {
                float4 val = reinterpret_cast<const float4*>(input)[global_idx];
                reinterpret_cast<float4*>(shared_input)[shared_idx] = val;
            } else if (global_idx < n) {
                // Handle remaining elements
                for (int j = 0; j < 4 && (global_idx*4 + j) < n && (shared_idx*4 + j) < TILE_SIZE; j++) {
                    shared_input[shared_idx*4 + j] = input[global_idx*4 + j];
                }
            }
        }
        
        __syncthreads();
        
        // Process data from shared memory
        for (int i = 0; i < FLOAT4_PER_THREAD; i++) {
            int shared_idx = tid * FLOAT4_PER_THREAD + i;
            int global_idx = tile_start/TILE_SIZE * (TILE_SIZE/4) + tid * FLOAT4_PER_THREAD + i;
            
            if (tile_start + shared_idx*4 < n && shared_idx*4 + 3 < TILE_SIZE) {
                // Process 4 elements as float4
                float4 in_vec = reinterpret_cast<float4*>(shared_input)[shared_idx];
                float4 out_vec;
                
                out_vec.x = (in_vec.x > 0.0f) ? in_vec.x : in_vec.x * negative_slope;
                out_vec.y = (in_vec.y > 0.0f) ? in_vec.y : in_vec.y * negative_slope;
                out_vec.z = (in_vec.z > 0.0f) ? in_vec.z : in_vec.z * negative_slope;
                out_vec.w = (in_vec.w > 0.0f) ? in_vec.w : in_vec.w * negative_slope;
                
                reinterpret_cast<float4*>(output + tile_start)[shared_idx] = out_vec;
            } else {
                // Handle remaining elements individually
                for (int j = 0; j < 4 && tile_start + shared_idx*4 + j < n && shared_idx*4 + j < TILE_SIZE; j++) {
                    float val = shared_input[shared_idx*4 + j];
                    output[tile_start + shared_idx*4 + j] = (val > 0.0f) ? val : val * negative_slope;
                }
            }
        }
        
        __syncthreads();
    }
}

// Improved version with better memory coalescing and simplified logic
__global__ void leaky_relu_optimized_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           float negative_slope,
                                           size_t n) {
    // Calculate global thread index
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // Process 4 elements per thread using float4
    if (idx + 3 < n) {
        //Aligned access - vectorized load/store
        float4 in_vec = reinterpret_cast<const float4*>(input)[idx / 4];
        float4 out_vec;
        
        out_vec.x = (in_vec.x > 0.0f) ? in_vec.x : in_vec.x * negative_slope;
        out_vec.y = (in_vec.y > 0.0f) ? in_vec.y : in_vec.y * negative_slope;
        out_vec.z = (in_vec.z > 0.0f) ? in_vec.z : in_vec.z * negative_slope;
        out_vec.w = (in_vec.w > 0.0f) ? in_vec.w : in_vec.w * negative_slope;
        
        reinterpret_cast<float4*>(output)[idx / 4] = out_vec;
    } else {
        // Handle remaining elements
        for (int i = 0; i < 4 && idx + i < n; ++i) {
            float val = input[idx + i];
            output[idx + i] = (val > 0.0f) ? val : val * negative_slope;
        }
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    
    // Use optimized kernel with better memory coalescing
    const int threads = 256;
    const int blocks = (n / 4 + threads - 1) / threads;
    
    // Limit blocks to avoid too many blocks for very large tensors
    const int max_blocks = 65535;
    const int effective_blocks = min(blocks, max_blocks);
    
    leaky_relu_optimized_kernel<<<effective_blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        n
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_forward, "Optimized Leaky ReLU forward with vectorization");
}
"""

leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized functional_model using vectorized CUDA kernel with improved memory coalescing.
    """
    # Ensure input is contiguous and float32
    if not x.is_contiguous():
        x = x.contiguous()
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    
    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    return output

# --- Constants and helper functions as required ---
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Use float32 to ensure compatibility with our custom CUDA kernel
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
