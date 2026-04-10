# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_150526/code_4.py
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
    extern __shared__ float shared_data[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float thread_sum = 0.0f;
    
    // Grid-stride loop ensures coalesced access
    for (int i = idx; i < n_elements; i += stride) {
        float diff = predictions[i] - targets[i];
        thread_sum += diff * diff;
    }
    
    // Block-level reduction
    shared_data[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, shared_data[0]);
    }
}

void mse_fused_forward(
    const torch::Tensor& predictions,
    const torch::Tensor& targets,
    torch::Tensor& output
) {
    const float* pred_ptr = predictions.data_ptr<float>();
    const float* target_ptr = targets.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    int n_elements = predictions.numel();
    
    // Initialize output to zero
    cudaMemset(output_ptr, 0, sizeof(float));
    
    int threads = 512;
    int blocks = 1024;
    
    mse_fused_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        pred_ptr, target_ptr, output_ptr, n_elements
    );
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void mse_fused_forward(
    const torch::Tensor& predictions,
    const torch::Tensor& targets,
    torch::Tensor& output
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
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(predictions, targets):
    # Ensure inputs are contiguous float32
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    output = torch.zeros(1, device=predictions.device, dtype=torch.float32)
    
    # Execute the fused kernel
    fused_mse_ext.mse_fused(predictions, targets, output)
    
    # Divide by count to get MSE
    return output / float(n_elements)

batch_size = 32768
input_shape = (32768,)

def get_init_inputs():
    return []

def get_inputs():
    scale = torch.rand(())
    return [torch.rand(batch_size, *input_shape).cuda()*scale, torch.rand(batch_size, *input_shape).cuda()]
