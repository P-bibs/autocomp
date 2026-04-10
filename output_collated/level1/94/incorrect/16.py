# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_151305/code_0.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void mse_loss_fused_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int64_t numel
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    
    float local_sum = 0.0f;
    for (int64_t i = idx; i < numel; i += stride) {
        float diff = predictions[i] - targets[i];
        local_sum += diff * diff;
    }
    
    // Use shared memory for block-level reduction
    __shared__ float sdata[1024];  // Assuming max 1024 threads per block
    int tid = threadIdx.x;
    sdata[tid] = local_sum;
    __syncthreads();
    
    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write block result to global memory
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

void mse_loss_fused_forward(int64_t numel, const float* predictions, const float* targets, float* output) {
    const int threads_per_block = 1024;
    const int blocks = min((numel + threads_per_block - 1) / threads_per_block, (int64_t)65535);
    
    // Initialize output to zero
    cudaMemset(output, 0, sizeof(float));
    
    mse_loss_fused_kernel<<<blocks, threads_per_block>>>(predictions, targets, output, numel);
    cudaDeviceSynchronize();
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function in the .cu file
void mse_loss_fused_forward(int64_t numel, const float* predictions, const float* targets, float* output);

torch::Tensor fused_mse_loss(torch::Tensor predictions, torch::Tensor targets) {
    const auto numel = predictions.numel();
    auto output = torch::zeros({}, predictions.options().dtype(torch::kFloat32));
    
    mse_loss_fused_forward(
        numel,
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>()
    );
    
    return output / static_cast<float>(numel);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_mse_loss", &fused_mse_loss, "Fused MSE Loss forward");
}
"""

# Compile the extension
fused_mse_ext = load_inline(
    name='fused_mse_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=compute_75', '-code=sm_75'],
    with_cuda=True
)

def functional_model(predictions, targets):
    return fused_mse_ext.fused_mse_loss(predictions, targets)

batch_size = 32768
input_shape = (32768,)

def get_init_inputs():
    return []

def get_inputs():
    scale = torch.rand(())
    return [torch.rand(batch_size, *input_shape)*scale, torch.rand(batch_size, *input_shape)]
