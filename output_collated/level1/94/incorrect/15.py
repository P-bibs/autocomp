# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_150526/code_5.py
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

# CUDA kernel to compute Mean Squared Error in a single pass
# This avoids intermediate allocations and multiple kernel launches.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mse_kernel(const float* __restrict__ preds, const float* __restrict__ targets, float* __restrict__ output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    // Grid-stride loop to handle sizes larger than thread count
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float diff = preds[i] - targets[i];
        sum += diff * diff;
    }

    sdata[tid] = sum;
    __syncthreads();

    // Block-level reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Atomic addition to global result
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

float mse_compute(torch::Tensor preds, torch::Tensor targets) {
    const int n = preds.numel();
    auto output = torch::zeros({1}, preds.options());
    
    // Launch configuration
    int threads = 512;
    int blocks = std::min(1024, (n + threads - 1) / threads);

    mse_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        preds.data_ptr<float>(), 
        targets.data_ptr<float>(), 
        output.data_ptr<float>(), 
        n
    );
    
    // Return mean by dividing sum by number of elements
    return output.item<float>() / static_cast<float>(n);
}
"""

cpp_source = r"""
#include <torch/extension.h>

float mse_compute(torch::Tensor preds, torch::Tensor targets);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mse_compute", &mse_compute, "Compute MSE in a single fused kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='mse_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(predictions, targets):
    """
    Optimized MSE implementation using a custom CUDA kernel.
    Ensures input is contiguous for linear memory access.
    """
    return fused_ext.mse_compute(predictions.contiguous(), targets.contiguous())

# Boilerplate for evaluation
batch_size = 32768
input_shape = (32768,)

def get_init_inputs():
    return []

def get_inputs():
    scale = torch.rand(())
    # Ensure tensors are on GPU
    return [
        (torch.rand(batch_size, *input_shape) * scale).cuda(), 
        torch.rand(batch_size, *input_shape).cuda()
    ]
