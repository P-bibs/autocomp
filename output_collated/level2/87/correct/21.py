# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_144040/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'subtract_value_1', 'subtract_value_2']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'subtract_value_1', 'subtract_value_2']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, subtracts two values, applies Mish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

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
    # State for conv (nn.Conv2d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
    if 'subtract_value_1' in flat_state:
        state_kwargs['subtract_value_1'] = flat_state['subtract_value_1']
    else:
        state_kwargs['subtract_value_1'] = getattr(model, 'subtract_value_1')
    if 'subtract_value_2' in flat_state:
        state_kwargs['subtract_value_2'] = flat_state['subtract_value_2']
    else:
        state_kwargs['subtract_value_2'] = getattr(model, 'subtract_value_2')
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
#include <cmath>

#define TILE_SIZE 16

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight,
    const float* __restrict__ bias, 
    float* __restrict__ output,
    int batch, int in_c, int in_h, int in_w,
    int out_c, int k, float sub1, float sub2) {
    
    int out_h_val = in_h - k + 1;
    int out_w_val = in_w - k + 1;
    
    // Calculate output indices
    int batch_idx = blockIdx.z;
    int out_c_idx = blockIdx.y;
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch || out_c_idx >= out_c) return;
    
    int out_h = spatial_idx / out_w_val;
    int out_w = spatial_idx % out_w_val;
    
    if (out_h >= out_h_val || out_w >= out_w_val) return;
    
    // Calculate output index in linear array
    int out_idx = ((batch_idx * out_c + out_c_idx) * out_h_val + out_h) * out_w_val + out_w;
    
    // Load bias
    float acc = bias[out_c_idx];
    
    // Weight pointer for this output channel - weights are [out_c, k, k, in_c]
    const float* weight_ptr = weight + (out_c_idx * k * k * in_c);
    
    // Calculate base input pointer for this spatial location
    const float* input_base = input + (batch_idx * in_c * in_h * in_w);
    
    // Perform convolution
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            int input_h = out_h + i;
            int input_w = out_w + j;
            
            if (input_h < in_h && input_w < in_w) {
                // Process all input channels for this kernel position
                const float* input_ptr = input_base + (input_h * in_w + input_w);
                
                // Vectorized load where possible - process 4 channels at once
                int ic = 0;
                for (; ic <= in_c - 4; ic += 4) {
                    float i0 = input_ptr[ic * in_h * in_w];
                    float i1 = input_ptr[(ic+1) * in_h * in_w];
                    float i2 = input_ptr[(ic+2) * in_h * in_w];
                    float i3 = input_ptr[(ic+3) * in_h * in_w];
                    
                    float w0 = weight_ptr[((i * k + j) * in_c + ic)];
                    float w1 = weight_ptr[((i * k + j) * in_c + ic + 1)];
                    float w2 = weight_ptr[((i * k + j) * in_c + ic + 2)];
                    float w3 = weight_ptr[((i * k + j) * in_c + ic + 3)];
                    
                    acc += i0 * w0 + i1 * w1 + i2 * w2 + i3 * w3;
                }
                
                // Handle remaining channels
                for (; ic < in_c; ++ic) {
                    acc += input_ptr[ic * in_h * in_w] * weight_ptr[((i * k + j) * in_c + ic)];
                }
            }
        }
    }
    
    // Apply Mish activation: x * tanh(ln(1 + e^x))
    float val = acc - sub1 - sub2;
    output[out_idx] = val * tanhf(logf(1.0f + expf(val)));
}

void fused_conv_mish(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    torch::Tensor output, 
    float sub1, 
    float sub2) {
    
    int batch = input.size(0);
    int in_c = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_c = weight.size(0);
    int k = weight.size(2);
    int out_h = in_h - k + 1;
    int out_w = in_w - k + 1;
    
    // Configure grid and block dimensions for better memory coalescing
    dim3 blockDim(256);
    dim3 gridDim(
        (out_h * out_w + blockDim.x - 1) / blockDim.x,  // Spatial dimensions
        out_c,                                          // Output channels
        batch                                           // Batch dimension
    );
    
    fused_conv_mish_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(),
        output.data_ptr<float>(), 
        batch, in_c, in_h, in_w, out_c, k, sub1, sub2
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                     torch::Tensor output, float sub1, float sub2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &fused_conv_mish, "Fused Convolution + Mish");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=0, 
                     conv_dilation=1, conv_groups=1, subtract_value_1, subtract_value_2):
    # Prepare weights for coalesced access: [out_c, k, k, in_c]
    w_reordered = conv_weight.permute(0, 2, 3, 1).contiguous()
    
    batch, _, h, w = x.shape
    k = conv_weight.shape[2]
    out = torch.empty((batch, conv_weight.size(0), h - k + 1, w - k + 1), device=x.device)
    
    fused_ext.fused_conv(x, w_reordered, conv_bias, out, subtract_value_1, subtract_value_2)
    return out
