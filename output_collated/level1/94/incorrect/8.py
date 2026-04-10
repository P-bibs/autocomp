# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_145724/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['predictions', 'targets']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks.

    Parameters:
        None
    """

    def __init__(self):
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

# CUDA kernel code
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void mse_kernel(const float* __restrict__ predictions,
                           const float* __restrict__ targets,
                           float* __restrict__ block_sums,
                           const long long N) {
    __shared__ float sdata[BLOCK_SIZE];
    
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = gridDim.x * blockDim.x;
    
    float thread_sum = 0.0f;
    
    // Grid-stride loop for coalesced access and handling large arrays
    for (long long i = tid; i < N; i += stride) {
        float diff = predictions[i] - targets[i];
        thread_sum += diff * diff;
    }
    
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_kernel(const float* __restrict__ block_sums,
                              float* __restrict__ total_sum,
                              const int num_blocks) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < num_blocks) ? block_sums[idx] : 0.0f;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(total_sum, sdata[0]);
    }
}
"""

# C++ bindings
cpp_source = r"""
#include <torch/extension.h>

void mse_cuda(const torch::Tensor& predictions, const torch::Tensor& targets, 
              torch::Tensor& block_sums, torch::Tensor& total_sum);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mse_cuda", &mse_cuda, "MSE CUDA kernel");
}
"""

# Custom MSE forward function
mse_cuda_code = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void mse_kernel(const float* __restrict__ predictions,
                           const float* __restrict__ targets,
                           float* __restrict__ block_sums,
                           const long long N) {
    __shared__ float sdata[BLOCK_SIZE];
    
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = gridDim.x * blockDim.x;
    
    float thread_sum = 0.0f;
    
    // Grid-stride loop for coalesced access and handling large arrays
    for (long long i = tid; i < N; i += stride) {
        float diff = predictions[i] - targets[i];
        thread_sum += diff * diff;
    }
    
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_kernel(const float* __restrict__ block_sums,
                              float* __restrict__ total_sum,
                              const int num_blocks) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < num_blocks) ? block_sums[idx] : 0.0f;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(total_sum, sdata[0]);
    }
}

void mse_cuda(const torch::Tensor& predictions, const torch::Tensor& targets, 
              torch::Tensor& block_sums, torch::Tensor& total_sum) {
    const long long N = predictions.numel();
    
    // Calculate grid size for maximum occupancy
    int block_size = BLOCK_SIZE;
    int grid_size = min((int)((N + block_size - 1) / block_size), 2048); // Cap at 2048 blocks
    
    // First kernel: compute block sums
    mse_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_sums.data_ptr<float>(),
        N
    );
    
    // Initialize total sum to zero
    cudaMemset(total_sum.data_ptr<float>(), 0, sizeof(float));
    
    // Second kernel: reduce block sums to final sum
    int reduce_grid = (grid_size + block_size - 1) / block_size;
    reduce_grid = max(reduce_grid, 1);
    reduce_kernel<<<reduce_grid, block_size>>>(
        block_sums.data_ptr<float>(),
        total_sum.data_ptr<float>(),
        grid_size
    );
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=mse_cuda_code,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(predictions, targets):
    # Ensure inputs are on GPU and flattened
    if predictions.device.type != 'cuda':
        predictions = predictions.cuda()
    if targets.device.type != 'cuda':
        targets = targets.cuda()
        
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    batch_size = predictions.numel()
    
    # Allocate intermediate tensors
    grid_size = min((batch_size + 256 - 1) // 256, 2048)  # Cap at 2048 blocks
    block_sums = torch.zeros(grid_size, dtype=torch.float32, device='cuda')
    total_sum = torch.zeros(1, dtype=torch.float32, device='cuda')
    
    # Call the custom CUDA kernel
    fused_ext.mse_cuda(predictions, targets, block_sums, total_sum)
    
    # Compute mean
    return total_sum.item() / batch_size

# Original helper functions for compatibility
def get_init_inputs():
    return []

def get_inputs():
    scale = torch.rand(()).cuda()
    return [torch.rand(32768, 32768).cuda()*scale, torch.rand(32768, 32768).cuda()]
