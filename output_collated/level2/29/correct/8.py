# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_104237/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['linear_weight', 'linear_bias']
REQUIRED_FLAT_STATE_NAMES = ['linear_weight', 'linear_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Mish, and applies Mish again.
    """

    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

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
    # State for linear (nn.Linear)
    if 'linear_weight' in flat_state:
        state_kwargs['linear_weight'] = flat_state['linear_weight']
    else:
        state_kwargs['linear_weight'] = getattr(model.linear, 'weight', None)
    if 'linear_bias' in flat_state:
        state_kwargs['linear_bias'] = flat_state['linear_bias']
    else:
        state_kwargs['linear_bias'] = getattr(model.linear, 'bias', None)
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- CUDA Kernel Code ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float softplus(float x) {
    // Use fast math equivalent for softplus
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ __forceinline__ float mish(float x) {
    // mish(x) = x * tanh(softplus(x))
    float sp = softplus(x);
    float th = tanhf(sp);
    return x * th;
}

__global__ void fused_linear_double_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int in_features,
    const int out_features,
    const int batch_size
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    if (batch_idx >= batch_size) return;

    const int out_elem = blockIdx.y * block_size + tid;
    if (out_elem >= out_features) return;

    const float* x = input + batch_idx * in_features;
    const float* w = weight + out_elem * in_features;
    
    // Compute linear: y = x * W[out_elem] + b[out_elem]
    float sum = 0.0f;
    for (int i = 0; i < in_features; ++i) {
        sum += x[i] * w[i];
    }
    sum += bias[out_elem];

    // Apply two sequential Mish activations
    sum = mish(sum);
    sum = mish(sum);

    output[batch_idx * out_features + out_elem] = sum;
}

void fused_linear_double_mish(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    // Launch configuration
    const int threads_per_block = 256;
    const dim3 block_dim(threads_per_block);
    const dim3 grid_dim(batch_size, (out_features + threads_per_block - 1) / threads_per_block);

    fused_linear_double_mish_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        in_features,
        out_features,
        batch_size
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_linear_double_mish(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_double_mish", &fused_linear_double_mish, "Fused Linear + Double Mish");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_linear_double_mish_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    linear_weight,
    linear_bias,
):
    # Create output tensor with the correct shape and device
    output = torch.empty(x.shape[0], linear_weight.shape[0], dtype=x.dtype, device=x.device)
    
    # Call the fused kernel
    fused_ext.fused_linear_double_mish(x, linear_weight, linear_bias, output)
    
    return output

batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda')]
