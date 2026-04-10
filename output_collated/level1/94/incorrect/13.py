# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_150526/code_0.py
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

# CUDA kernel for fused MSE computation with memory coalescing
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mse_fused_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n_elements
) {
    // Grid-stride loop for better memory coalescing
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float thread_sum = 0.0f;
    
    // Process multiple elements per thread to improve arithmetic intensity
    for (int i = idx; i < n_elements; i += stride) {
        float diff = predictions[i] - targets[i];
        thread_sum += diff * diff;
    }
    
    // Use shared memory for block-level reduction
    __shared__ float shared_data[1024]; // Assuming max 1024 threads per block
    int tid = threadIdx.x;
    shared_data[tid] = thread_sum;
    __syncthreads();
    
    // Block-level reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    
    // Write block result to global memory
    if (tid == 0) {
        atomicAdd(output, shared_data[0]);
    }
}

void mse_fused_forward(
    const torch::Tensor& predictions,
    const torch::Tensor& targets,
    torch::Tensor& output,
    const int blocks,
    const int threads
) {
    const float* pred_ptr = predictions.data_ptr<float>();
    const float* target_ptr = targets.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    int n_elements = predictions.numel();
    
    // Initialize output to zero
    cudaMemset(output_ptr, 0, sizeof(float));
    
    mse_fused_kernel<<<blocks, threads>>>(pred_ptr, target_ptr, output_ptr, n_elements);
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void mse_fused_forward(
    const torch::Tensor& predictions,
    const torch::Tensor& targets,
    torch::Tensor& output,
    const int blocks,
    const int threads
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mse_fused", &mse_fused_forward, "Fused MSE forward pass");
}
"""

# Compile the extension
fused_mse_ext = load_inline(
    name='fused_mse',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=compute_75', '-code=sm_75'],
    with_cuda=True
)

def functional_model(predictions, targets):
    # Determine optimal grid and block dimensions
    n_elements = predictions.numel()
    threads_per_block = 1024  # Max threads per block for good occupancy
    blocks = min(65535, (n_elements + threads_per_block - 1) // threads_per_block)  # Max grid size
    
    # Allocate output tensor
    output = torch.zeros(1, device=predictions.device, dtype=torch.float32)
    
    # Call fused CUDA kernel
    fused_mse_ext.mse_fused(predictions, targets, output, blocks, threads_per_block)
    
    # Divide by number of elements to get mean
    return output / n_elements

batch_size = 32768
input_shape = (32768,)

def get_init_inputs():
    return []

def get_inputs():
    scale = torch.rand(())
    return [torch.rand(batch_size, *input_shape)*scale, torch.rand(batch_size, *input_shape)]
