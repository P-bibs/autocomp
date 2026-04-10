# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_140827/code_1.py
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

# Optimization: Fuse subtraction, power, and mean into a single kernel
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void mse_kernel(const float* pred, const float* target, float* result, size_t n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    while (i < n) {
        float diff = pred[i] - target[i];
        sum += diff * diff;
        i += blockDim.x * gridDim.x;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(result, sdata[0]);
}

void fused_mse_forward(torch::Tensor predictions, torch::Tensor targets, torch::Tensor output) {
    size_t n = predictions.numel();
    
    int threads = 256;
    int blocks = min(1024, (int)((n + threads - 1) / threads));
    
    mse_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        predictions.data_ptr<float>(), 
        targets.data_ptr<float>(), 
        output.data_ptr<float>(), 
        n
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_mse_forward(torch::Tensor predictions, torch::Tensor targets, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_mse_forward", &fused_mse_forward, "Fused MSE forward pass");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_mse',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

class FusedMSEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predictions, targets):
        # Ensure inputs are contiguous float32 for the custom kernel
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        
        n = predictions.numel()
        result = torch.zeros(1, device=predictions.device, dtype=predictions.dtype)
        
        fused_ext.fused_mse_forward(predictions, targets, result)
        
        return result / n

def functional_model(predictions, targets):
    return FusedMSEFunction.apply(predictions, targets)

# Boilerplate for evaluation environment
batch_size = 32768
input_shape = (32768,)

def get_init_inputs():
    return []

def get_inputs():
    scale = torch.rand(()).cuda()
    return [torch.rand(batch_size, *input_shape).cuda()*scale, torch.rand(batch_size, *input_shape).cuda()]
