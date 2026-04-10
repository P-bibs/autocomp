# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_145724/code_2.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- CUDA Kernel Code ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_mse_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ result,
    const int64_t numel
) {
    // Use shared memory for block-level reduction
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop to handle all elements
    float sum = 0.0f;
    for (int64_t i = idx; i < numel; i += blockDim.x * gridDim.x) {
        float diff = predictions[i] - targets[i];
        sum += diff * diff;
    }
    
    // Store thread's sum in shared memory
    sdata[tid] = sum;
    __syncthreads();
    
    // Block-level reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write block result to global memory with atomic add
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

void fused_mse_forward(
    const at::Tensor& predictions,
    const at::Tensor& targets,
    at::Tensor& result
) {
    // Ensure tensors are on CUDA
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "Only Float32 supported");
    TORCH_CHECK(targets.dtype() == torch::kFloat32, "Only Float32 supported");
    
    const int64_t numel = predictions.numel();
    TORCH_CHECK(targets.numel() == numel, "predictions and targets must have same number of elements");
    
    // Set device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(predictions));
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int blocks_per_grid = min((int64_t)65535, (numel + threads_per_block - 1) / threads_per_block);
    const size_t shared_mem_size = threads_per_block * sizeof(float);
    
    // Zero the result tensor before kernel launch
    result.zero_();
    
    // Launch kernel
    fused_mse_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        result.data_ptr<float>(),
        numel
    );
    
    // Check for kernel launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_mse_forward(const at::Tensor& predictions, const at::Tensor& targets, at::Tensor& result);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_mse_forward", &fused_mse_forward, "Fused MSE forward pass");
}
"""

# Compile the extension
fused_mse_ext = load_inline(
    name='fused_mse_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def fused_mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes MSE loss using a custom fused CUDA kernel."""
    # Validate inputs
    assert predictions.shape == targets.shape, "Input tensors must have the same shape"
    assert predictions.dtype == torch.float32, "Only Float32 tensors supported"
    assert targets.dtype == torch.float32, "Only Float32 tensors supported"
    
    # Create output tensor for the sum (result will be a single-element tensor)
    result_tensor = torch.zeros(1, dtype=torch.float32, device=predictions.device)
    
    # Call the custom CUDA kernel
    fused_mse_ext.fused_mse_forward(predictions, targets, result_tensor)
    
    # Compute mean by dividing the sum by the number of elements
    numel = predictions.numel()
    return result_tensor / numel

def functional_model(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Optimized version using fused CUDA kernel for MSE computation."""
    return fused_mse_loss(predictions, targets)

batch_size = 32768
input_shape = (32768,)

def get_init_inputs():
    return []

def get_inputs():
    scale = torch.rand(()).cuda()  # Move to GPU for consistency
    pred = torch.rand(batch_size, *input_shape, device='cuda') * scale
    targ = torch.rand(batch_size, *input_shape, device='cuda')
    return [pred, targ]
