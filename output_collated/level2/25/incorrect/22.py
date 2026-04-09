# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_083856/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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

# CUDA kernel for fused operation: conv2d + min reduction + tanh(tanh(x))
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int batch_size,
    int in_channels,
    int height,
    int width,
    int out_channels,
    int kernel_size
) {
    // Compute global thread indices
    int batch_idx = blockIdx.z;
    int out_row = blockIdx.y;
    int out_col = blockIdx.x;
    
    // Early exit if out of bounds
    if (batch_idx >= batch_size || out_row >= (height - kernel_size + 1) || out_col >= (width - kernel_size + 1)) {
        return;
    }
    
    float min_val = INFINITY;
    
    // Iterate over all output channels to compute conv + find min
    for (int oc = 0; oc < out_channels; ++oc) {
        float sum = bias[oc];
        
        // Perform convolution for this output channel
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int x_row = out_row + kh;
                    int x_col = out_col + kw;
                    
                    // Direct access without boundary checks as per problem assumption (padding=0)
                    float x_val = x[(((batch_idx * in_channels) + ic) * height + x_row) * width + x_col];
                    float w_val = weight[(((oc * in_channels) + ic) * kernel_size + kh) * kernel_size + kw];
                    sum += x_val * w_val;
                }
            }
        }
        
        // Update minimum value
        if (sum < min_val) {
            min_val = sum;
        }
    }
    
    // Apply double tanh (as in original) and write to output
    float result = tanhf(tanhf(min_val));
    int out_height = height - kernel_size + 1;
    int out_width = width - kernel_size + 1;
    out[(((batch_idx * 1) + 0) * out_height + out_row) * out_width + out_col] = result;
}

void fused_conv_min_tanh_forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& out
) {
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    // Output spatial dimensions after convolution with padding=0
    int out_height = height - kernel_size + 1;
    int out_width = width - kernel_size + 1;
    
    // Configure grid and block dimensions
    dim3 grid(out_width, out_height, batch_size);
    dim3 block(1, 1, 1); // One thread per output position
    
    // Launch kernel
    fused_conv_min_tanh_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        in_channels,
        height,
        width,
        out_channels,
        kernel_size
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function in the .cu file
void fused_conv_min_tanh_forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& out
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv_min_tanh_forward, "Fused Conv-Min-Tanh forward pass");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_min_tanh_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
):
    # Validate assumptions from the prompt
    assert conv_stride == 1, "Only stride=1 supported"
    assert conv_padding == 0, "Only padding=0 supported"
    assert conv_dilation == 1, "Only dilation=1 supported"
    assert conv_groups == 1, "Only groups=1 supported"
    assert conv_weight.size(2) == conv_weight.size(3), "Only square kernels supported"
    
    batch_size = x.size(0)
    in_channels = x.size(1)
    height = x.size(2)
    width = x.size(3)
    out_channels = conv_weight.size(0)
    kernel_size = conv_weight.size(2)
    
    # Output spatial dimensions
    out_height = height - kernel_size + 1
    out_width = width - kernel_size + 1
    
    # Allocate output tensor
    out = torch.empty((batch_size, 1, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Run fused kernel
    fused_ext.fused_op(x, conv_weight, conv_bias, out)
    
    return out

# Test parameters
batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

# Helper functions for testing (as in original)
def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
