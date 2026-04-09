# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_091623/code_13.py
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

# ============================================================================
# CUDA Kernel Implementation: Fused Conv2D + Min + Tanh
# ============================================================================
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding
) {
    int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    // Each thread calculates one spatial pixel for one batch element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = batch_size * out_h * out_w;
    
    if (idx >= total_pixels) return;
    
    int b = idx / (out_h * out_w);
    int remaining = idx % (out_h * out_w);
    int oh = remaining / out_w;
    int ow = remaining % out_w;
    
    float min_val = 1e30f; // Initialize with large value
    
    // Iterate over output channels
    for (int oc = 0; oc < out_channels; oc++) {
        float acc = bias[oc];
        
        // Sum over kernel and in_channels
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                int ih = oh * stride + kh - padding;
                if (ih < 0 || ih >= height) continue;
                
                for (int kw = 0; kw < kernel_size; kw++) {
                    int iw = ow * stride + kw - padding;
                    if (iw < 0 || iw >= width) continue;
                    
                    float val = input[((b * in_channels + ic) * height + ih) * width + iw];
                    float w = weight[((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw];
                    acc += val * w;
                }
            }
        }
        if (acc < min_val) min_val = acc;
    }
    
    output[idx] = tanhf(min_val);
}

void fused_conv_min_tanh(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (width + 2 * padding - kernel_size) / stride + 1;
    int total_pixels = batch_size * out_h * out_w;
    
    int threads = 256;
    int blocks = (total_pixels + threads - 1) / threads;
    
    fused_conv_min_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        height, width, kernel_size, stride, padding
    );
}
"""

cpp_source = r"""
void fused_conv_min_tanh(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_tanh", &fused_conv_min_tanh, "Fused conv/min/tanh kernel");
}
"""

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
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation, # Note: Kernel supports dilation=1 only as per requirements
    conv_groups,   # Note: Kernel supports groups=1 only
):
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    out_channels = conv_weight.shape[0]
    height, width = x.shape[2], x.shape[3]
    kernel_size = conv_weight.shape[2]
    
    out_h = (height + 2 * conv_padding - kernel_size) // conv_stride + 1
    out_w = (width + 2 * conv_padding - kernel_size) // conv_stride + 1
    
    output = torch.empty((batch_size, 1, out_h, out_w), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_conv_min_tanh(
        x.contiguous(), 
        conv_weight.contiguous(), 
        conv_bias.contiguous(), 
        output, 
        conv_stride, 
        conv_padding
    )
    return output
