# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_073421/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'subtract_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['linear_weight', 'linear_bias', 'subtract_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['linear_weight', 'linear_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a matrix multiplication, subtraction, multiplication, and ReLU activation.
    """

    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

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
    if 'subtract_value' in flat_state:
        state_kwargs['subtract_value'] = flat_state['subtract_value']
    else:
        state_kwargs['subtract_value'] = getattr(model, 'subtract_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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
#include <cuda_fp16.h>

__global__ void fused_linear_sub_mul_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float subtract_value,
    const float multiply_value,
    const int batch_size,
    const int in_features,
    const int out_features
) {
    const int batch_idx = blockIdx.x;
    const int out_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_idx >= out_features) return;
    
    const int input_offset = batch_idx * in_features;
    const int output_offset = batch_idx * out_features + out_idx;
    const int weight_offset = out_idx * in_features;
    
    float sum = 0.0f;
    
    // Perform dot product
    for (int i = 0; i < in_features; ++i) {
        sum += input[input_offset + i] * weight[weight_offset + i];
    }
    
    // Add bias
    sum += bias[out_idx];
    
    // Subtract constant
    sum -= subtract_value;
    
    // Multiply by constant
    sum *= multiply_value;
    
    // Apply ReLU
    sum = fmaxf(sum, 0.0f);
    
    // Write output
    output[output_offset] = sum;
}

void fused_linear_sub_mul_relu_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const float subtract_value,
    const float multiply_value,
    const int batch_size,
    const int in_features,
    const int out_features
) {
    const int threads_per_block = 256;
    const dim3 blocks(batch_size, (out_features + threads_per_block - 1) / threads_per_block);
    const dim3 threads(threads_per_block);
    
    fused_linear_sub_mul_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        subtract_value,
        multiply_value,
        batch_size,
        in_features,
        out_features
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_linear_sub_mul_relu_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const float subtract_value,
    const float multiply_value,
    const int batch_size,
    const int in_features,
    const int out_features
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_sub_mul_relu_forward", &fused_linear_sub_mul_relu_forward, "Fused Linear Sub Mul ReLU forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
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
    subtract_value,
    multiply_value,
):
    batch_size = x.size(0)
    in_features = x.size(1)
    out_features = linear_weight.size(0)
    
    output = torch.empty(batch_size, out_features, device=x.device, dtype=x.dtype)
    
    fused_ext.fused_linear_sub_mul_relu_forward(
        x.contiguous(),
        linear_weight.contiguous(),
        linear_bias.contiguous(),
        output,
        float(subtract_value),
        float(multiply_value),
        batch_size,
        in_features,
        out_features
    )
    
    return output

batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
