# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_141903/code_4.py
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
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused MSE computation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_mse_forward_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int num_elements
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    float local_sum = 0.0f;
    
    // Grid-stride loop for better GPU utilization
    for (int i = tid; i < num_elements; i += stride) {
        float diff = predictions[i] - targets[i];
        local_sum += diff * diff;
    }
    
    // Warp-level reduction using shuffle primitives
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    // Shared memory for block reduction
    extern __shared__ float sdata[];
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    if (lane_id == 0) {
        sdata[warp_id] = local_sum;
    }
    
    __syncthreads();
    
    // Final reduction in shared memory for the current block
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        local_sum = (lane_id < num_warps) ? sdata[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
        
        // Single atomic operation per block to update global result
        if (lane_id == 0) {
            atomicAdd(output, local_sum);
        }
    }
}

void fused_mse_forward(
    const at::Tensor& predictions,
    const at::Tensor& targets,
    at::Tensor& output,
    const int blocks,
    const int threads
) {
    const int num_elements = predictions.numel();
    const float* pred_ptr = predictions.data_ptr<float>();
    const float* target_ptr = targets.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    // Initialize output to zero using non-blocking API
    cudaMemset(output_ptr, 0, sizeof(float));
    
    // Shared memory size depends on the number of warps in a block
    int num_warps = (threads + 31) / 32;
    size_t shared_mem_size = num_warps * sizeof(float);
    
    fused_mse_forward_kernel<<<blocks, threads, shared_mem_size>>>(
        pred_ptr, target_ptr, output_ptr, num_elements
    );
}
"""

# C++ binding
cpp_source = r"""
#include <torch/extension.h>

void fused_mse_forward(
    const at::Tensor& predictions,
    const at::Tensor& targets,
    at::Tensor& output,
    const int blocks,
    const int threads
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_mse", &fused_mse_forward, "Fused MSE Forward");
}
"""

# Compile the extension inline
fused_mse_ext = load_inline(
    name='fused_mse',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(predictions, targets):
    num_elements = predictions.numel()
    
    # Kernel parameters
    threads_per_block = 256
    # Aim for enough occupancy; limited by hardware
    num_blocks = 1024 
    
    output = torch.zeros(1, device=predictions.device, dtype=torch.float)
    
    fused_mse_ext.fused_mse(predictions, targets, output, num_blocks, threads_per_block)
    
    return output / float(num_elements)

# Setup globals for environment
batch_size = 32768
input_shape = (32768,)

def get_init_inputs():
    return []

def get_inputs():
    scale = torch.rand(()).cuda()
    # Ensure inputs are contiguous float tensors for the kernel
    p = (torch.rand(batch_size, *input_shape, device='cuda') * scale).contiguous()
    t = (torch.rand(batch_size, *input_shape, device='cuda')).contiguous()
    return [p, t]
