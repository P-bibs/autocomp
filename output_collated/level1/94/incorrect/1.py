# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_140827/code_5.py
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

# Optimization: Fuse subtraction, power, and mean reduction into a single kernel.
# We use shared memory reduction to minimize atomic contention and global memory writes.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mse_kernel(const float* __restrict__ pred, const float* __restrict__ target, float* result, size_t n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    // Grid-stride loop to handle data size larger than grid
    for (size_t i = gid; i < n; i += blockDim.x * gridDim.x) {
        float diff = pred[i] - target[i];
        sum += diff * diff;
    }

    sdata[tid] = sum;
    __syncthreads();

    // Reduce shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

float fused_mse(torch::Tensor predictions, torch::Tensor targets) {
    size_t n = predictions.numel();
    auto result = torch::zeros({1}, predictions.options());
    
    const int threads = 512;
    const int blocks = std::min((int)((n + threads - 1) / threads), 1024);
    
    mse_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        predictions.data_ptr<float>(), 
        targets.data_ptr<float>(), 
        result.data_ptr<float>(), 
        n
    );
    
    // Return mean: sum / n
    return result.item<float>() / static_cast<float>(n);
}
"""

cpp_source = r"""
#include <torch/extension.h>

float fused_mse(torch::Tensor predictions, torch::Tensor targets);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_mse", &fused_mse, "Fused Mean Squared Error kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_mse_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(predictions, targets):
    # The custom kernel expects contiguous GPU tensors
    return fused_ext.fused_mse(predictions.contiguous(), targets.contiguous())

# Boilerplate for evaluation
batch_size = 32768
input_shape = (32768,)

def get_init_inputs():
    return []

def get_inputs():
    # Scale to ensure numerical variety
    scale = torch.rand(())
    return [
        (torch.rand(batch_size * input_shape[0], device='cuda') * scale),
        (torch.rand(batch_size * input_shape[0], device='cuda'))
    ]
