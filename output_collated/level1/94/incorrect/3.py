# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_141903/code_1.py
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

# Optimization: Fusing operations (Subtraction, Power, Accumulation) into one kernel
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_mse_kernel(const float* __restrict__ preds, 
                                 const float* __restrict__ targets, 
                                 float* result, 
                                 size_t n) {
    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        float diff = preds[i] - targets[i];
        sum += diff * diff;
    }
    
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(result, sdata[0]);
}

void fused_mse_forward(torch::Tensor preds, torch::Tensor targets, torch::Tensor result) {
    size_t n = preds.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = (blocks > 1024) ? 1024 : blocks;

    fused_mse_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        preds.data_ptr<float>(), targets.data_ptr<float>(), result.data_ptr<float>(), n
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function in the .cu file
void fused_mse_forward(torch::Tensor preds, torch::Tensor targets, torch::Tensor result);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_mse", &fused_mse_forward, "Fused MSE computation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_mse_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(predictions, targets):
    # Ensure inputs are contiguous and on GPU
    preds = predictions.contiguous().to(torch.float32)
    tgts = targets.contiguous().to(torch.float32)
    
    # Create output tensor
    result = torch.zeros(1, device=preds.device, dtype=torch.float32)
    
    # Call fused kernel
    fused_ext.fused_mse(preds, tgts, result)
    
    # Return mean
    return result / predictions.numel()

# Boilerplate for evaluation environment
batch_size = 32768
input_shape = (32768,)

def get_init_inputs(): 
    return []

def get_inputs():
    scale = torch.rand(()).to('cuda')
    return [torch.rand((batch_size,), device='cuda')*scale, torch.rand((batch_size,), device='cuda')]
