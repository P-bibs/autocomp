# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_020747/code_5.py
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

# Define the CUDA kernel for fused operation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_l1_normalization_kernel(
    const float* input,
    float* output,
    const int batch_size,
    const int dim
) {
    // Each block processes one row
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    // Shared memory for reduction
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Initialize shared memory
    sdata[tid] = 0.0f;
    
    // Grid-stride loop to accumulate absolute values
    const float* row_input = input + row * dim;
    float sum = 0.0f;
    for (int i = tid; i < dim; i += stride) {
        sum += fabsf(row_input[i]);
    }
    sdata[tid] = sum;
    
    __syncthreads();
    
    // Warp-level reduction
    for (int s = 16; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 in each block computes the mean and stores it
    float mean = sdata[0] / dim;
    if (mean == 0.0f) mean = 1.0f; // Avoid division by zero
    
    // Grid-stride loop to compute output
    float* row_output = output + row * dim;
    for (int i = tid; i < dim; i += stride) {
        row_output[i] = row_input[i] / mean;
    }
}

void fused_l1_normalization_forward(
    const at::Tensor& input,
    at::Tensor& output,
    const int batch_size,
    const int dim
) {
    // Set CUDA device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    // Launch configuration
    const int threads_per_block = 1024;
    const int blocks = batch_size;
    const int shared_mem_size = threads_per_block * sizeof(float);
    
    // Launch kernel
    fused_l1_normalization_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA error in fused_l1_normalization_kernel: ", cudaGetErrorString(err));
    }
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_l1_normalization_forward(
    const at::Tensor& input,
    at::Tensor& output,
    const int batch_size,
    const int dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_l1_normalization", &fused_l1_normalization_forward, "Fused L1 normalization forward pass");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_l1_normalization_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """Optimized implementation using custom CUDA kernel"""
    # Ensure input is the right type and on GPU
    if not x.is_cuda:
        x = x.cuda()
    if x.dtype != torch.float32:
        x = x.float()
    
    # Create output tensor with same properties as input
    output = torch.empty_like(x)
    
    # Extract dimensions
    batch_size, dim = x.shape
    
    # Call custom CUDA kernel
    fused_ext.fused_l1_normalization(x, output, batch_size, dim)
    
    return output

# For testing purposes
batch_size = 32768
dim = 65535

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]
