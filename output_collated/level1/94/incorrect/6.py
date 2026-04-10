# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_141903/code_7.py
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

# =============================================================================
# 1. CUDA-kernel source
# Using a grid-strided loop to handle arbitrary N and shared memory reduction
# to minimize global memory traffic.
# =============================================================================
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

__global__ void mse_sum_kernel(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    const int64_t N,
    float* __restrict__ output)
{
    // Shared memory for block-wise reduction
    extern __shared__ float sdata[];

    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    float acc = 0.0f;

    // Grid-strided loop to handle N > num_threads
    for (int64_t i = tid; i < N; i += stride) {
        float diff = pred[i] - target[i];
        acc += diff * diff;
    }

    sdata[threadIdx.x] = acc;
    __syncthreads();

    // Block-wise reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Atomic add to global memory for the block result
    if (threadIdx.x == 0) {
        atomicAdd(output, sdata[0]);
    }
}

void launch_mse_sum(torch::Tensor pred, torch::Tensor target, torch::Tensor output) {
    const int64_t N = pred.numel();
    const int threads = 256;
    // Aim for enough blocks to saturate the GPU
    const int blocks = 1024; 

    mse_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        pred.data_ptr<float>(),
        target.data_ptr<float>(),
        N,
        output.data_ptr<float>()
    );
}
"""

# =============================================================================
# 2. C++ Binding, Compilation, and Wrapper
# =============================================================================
cpp_source = r"""
#include <torch/extension.h>
void launch_mse_sum(torch::Tensor pred, torch::Tensor target, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_mse_sum", &launch_mse_sum, "Fused MSE Sum Kernel");
}
"""

fused_ext = load_inline(
    name='mse_fused',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(predictions, targets):
    """
    Optimized functional_model using a fused CUDA kernel.
    Reduces global memory complexity from O(3N) writes to O(1) per block.
    """
    # Ensure tensors are contiguous and on GPU
    pred = predictions.contiguous()
    tgt = targets.contiguous()
    
    # Pre-allocate output buffer for the atomic add
    sum_out = torch.zeros(1, dtype=torch.float32, device=pred.device)
    
    # Launch fused kernel
    fused_ext.launch_mse_sum(pred, tgt, sum_out)
    
    # Return MSE
    return sum_out / pred.numel()

# Boilerplate for the specific execution context
batch_size = 32768
input_shape = (32768,)

def get_init_inputs():
    return []

def get_inputs():
    # Helper to generate inputs similar to the original requirement
    scale = torch.rand(())
    return [
        (torch.rand(batch_size, *input_shape) * scale).cuda(), 
        torch.rand(batch_size, *input_shape).cuda()
    ]
