# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110448/code_6.py
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
from torch.utils.cpp_extension import load_inline

# The CUDA kernel performs:
# 1. Linear layer (y = xW^T + b)
# 2. Mish activation (mish(x) = x * tanh(ln(1 + exp(x))))
# 3. Repeat Mish activation
# We treat this as a row-major matrix-vector operation per output element for efficiency.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ inline float mish(float x) {
    // x * tanh(softplus(x))
    // softplus(x) = ln(1 + exp(x))
    return x * tanhf(logf(1.0f + expf(x > 20.0f ? 20.0f : x)));
}

__global__ void fused_mish_linear_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // output row
    int j = blockIdx.y * blockDim.y + threadIdx.y; // output col

    if (i < batch_size && j < out_features) {
        float acc = bias[j];
        for (int k = 0; k < in_features; ++k) {
            acc += input[i * in_features + k] * weight[j * in_features + k];
        }
        
        float res = mish(acc);
        res = mish(res);
        
        output[i * out_features + j] = res;
    }
}

void fused_op_forward(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    torch::Tensor output) 
{
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    dim3 threads(16, 16);
    dim3 blocks((batch_size + threads.x - 1) / threads.x, (out_features + threads.y - 1) / threads.y);

    fused_mish_linear_kernel<<<blocks, threads>>>(
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

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Linear + 2x Mish activation");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias):
    batch_size, _ = x.shape
    out_features = linear_bias.shape[0]
    output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
    
    # Weight in PyTorch Linear is [out_features, in_features]
    fused_ext.fused_op(x, linear_weight, linear_bias, output)
    return output

# Helper variables for evaluation
batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda')]

# Note: The caller must provide weights/bias on CUDA device.
