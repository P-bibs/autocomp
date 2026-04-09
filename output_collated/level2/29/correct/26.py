# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110448/code_0.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_linear_double_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features) {
    
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_idx >= out_features) return;
    
    // Compute linear operation
    float sum = (bias != nullptr) ? bias[out_idx] : 0.0f;
    
    for (int i = 0; i < in_features; ++i) {
        sum += input[batch_idx * in_features + i] * weight[out_idx * in_features + i];
    }
    
    // Apply first Mish activation: x * tanh(softplus(x))
    float x = sum;
    float softplus = x > 20.0f ? x : (x < -20.0f ? 0.0f : logf(expf(x) + 1.0f));
    float mish1 = x * tanhf(softplus);
    
    // Apply second Mish activation
    x = mish1;
    softplus = x > 20.0f ? x : (x < -20.0f ? 0.0f : logf(expf(x) + 1.0f));
    float mish2 = x * tanhf(softplus);
    
    output[batch_idx * out_features + out_idx] = mish2;
}

void fused_linear_double_mish_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_features,
    int out_features) {
    
    dim3 block_size(256);
    dim3 grid_size(batch_size, (out_features + block_size.x - 1) / block_size.x);
    
    fused_linear_double_mish_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_linear_double_mish_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_features,
    int out_features);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_double_mish", &fused_linear_double_mish_forward, "Fused Linear + Double Mish forward pass");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_linear_double_mish',
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
    batch_size = x.shape[0]
    in_features = x.shape[1]
    out_features = linear_weight.shape[0]
    
    output = torch.empty(batch_size, out_features, device=x.device, dtype=x.dtype)
    fused_ext.fused_linear_double_mish(x, linear_weight, linear_bias, output, batch_size, in_features, out_features)
    return output

# Test functions (not used during evaluation but kept for compatibility)
def get_init_inputs():
    return [8192, 8192]

def get_inputs():
    return [torch.rand(1024, 8192, device='cuda')]
