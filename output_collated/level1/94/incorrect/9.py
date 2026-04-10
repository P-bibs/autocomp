# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_145724/code_1.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mse_kernel(const float* __restrict__ preds, const float* __restrict__ targets, float* output, size_t n) {
    extern __shared__ float shared_sums[];
    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    float thread_sum = 0.0f;

    // Grid-stride loop for processing all elements
    for (size_t i = gid; i < n; i += blockDim.x * gridDim.x) {
        float diff = preds[i] - targets[i];
        thread_sum += diff * diff;
    }

    shared_sums[tid] = thread_sum;
    __syncthreads();

    // Reduction in shared memory
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sums[tid] += shared_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared_sums[0]);
    }
}

void mse_forward_cuda(torch::Tensor predictions, torch::Tensor targets, torch::Tensor output) {
    size_t n = predictions.numel();
    
    int threads = 256;
    int blocks = min(1024, (int)((n + threads - 1) / threads));
    size_t shared_size = threads * sizeof(float);

    // Initialize output to zero
    cudaMemset(output.data_ptr<float>(), 0, sizeof(float));

    mse_kernel<<<blocks, threads, shared_size>>>(
        predictions.data_ptr<float>(), 
        targets.data_ptr<float>(), 
        output.data_ptr<float>(), 
        n
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void mse_forward_cuda(torch::Tensor predictions, torch::Tensor targets, torch::Tensor output);

torch::Tensor mse_forward(torch::Tensor predictions, torch::Tensor targets) {
    auto output = torch::zeros({1}, predictions.options());
    mse_forward_cuda(predictions, targets, output);
    return output / static_cast<float>(predictions.numel());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mse_loss", &mse_forward, "MSE Loss Kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='mse_loss',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(predictions, targets):
    return fused_ext.mse_loss(predictions, targets)

batch_size = 32768
input_shape = (32768,)

def get_init_inputs():
    return []

def get_inputs():
    scale = torch.rand(())
    return [torch.rand(batch_size, *input_shape)*scale, torch.rand(batch_size, *input_shape)]
