# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_145724/code_4.py
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

# CUDA kernel with shared memory reduction for maximal throughput
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void mse_fused_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const size_t numel
) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    // Grid-stride loop to handle problem sizes larger than grid capacity
    for (size_t i = idx; i < numel; i += blockDim.x * gridDim.x) {
        float diff = predictions[i] - targets[i];
        sum += diff * diff;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Shared memory reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Atomic accumulation from each block
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

void mse_fused_forward(torch::Tensor predictions, torch::Tensor targets, torch::Tensor output) {
    const size_t numel = predictions.numel();
    const int threads = 512;
    // Cap blocks to prevent grid oversubscription
    const int blocks = std::min((size_t)1024, (numel + threads - 1) / threads);
    
    mse_fused_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        numel
    );
}
"""

cpp_source = r"""
void mse_fused_forward(torch::Tensor predictions, torch::Tensor targets, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mse_fused", &mse_fused_forward, "Fused MSE computation");
}
"""

# Compile the extension just-in-time
fused_mse_ext = load_inline(
    name='fused_mse',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(predictions, targets):
    # Ensure inputs are contiguous float tensors on GPU
    predictions = predictions.contiguous().float()
    targets = targets.contiguous().float()
    
    # Pre-allocate output buffer
    output = torch.zeros(1, device=predictions.device, dtype=torch.float32)
    
    # Launch fused kernel
    fused_mse_ext.mse_fused(predictions, targets, output)
    
    # Final normalization
    return output / predictions.numel()

# Parameters based on original requirements
batch_size = 32768
input_shape = (32768,)

def get_init_inputs():
    return []

def get_inputs():
    # Use specified device if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scale = torch.rand((), device=device)
    return [torch.rand(batch_size, *input_shape, device=device)*scale, 
            torch.rand(batch_size, *input_shape, device=device)]
