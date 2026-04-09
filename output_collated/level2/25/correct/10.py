# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_090933/code_3.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 16

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int batch,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int out_height,
    int out_width
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.z;
    
    if (out_x >= out_width || out_y >= out_height) return;
    
    float min_val = INFINITY;
    
    // Iterate through all output channels to find minimum
    for (int oc = 0; oc < out_channels; oc++) {
        float sum = b[oc];
        
        // Convolution operation
        for (int ic = 0; ic < in_channels; ic++) {
            for (int ky = 0; ky < kernel_h; ky++) {
                for (int kx = 0; kx < kernel_w; kx++) {
                    int in_x = out_x + kx;
                    int in_y = out_y + ky;
                    
                    // Assuming padding=0, stride=1
                    if (in_x < width && in_y < height) {
                        int x_idx = ((batch_idx * in_channels + ic) * height + in_y) * width + in_x;
                        int w_idx = ((oc * in_channels + ic) * kernel_h + ky) * kernel_w + kx;
                        sum += x[x_idx] * w[w_idx];
                    }
                }
            }
        }
        
        if (sum < min_val) {
            min_val = sum;
        }
    }
    
    // Apply double tanh
    float result = tanhf(tanhf(min_val));
    
    // Write result
    int out_idx = ((batch_idx * 1 + 0) * out_height + out_y) * out_width + out_x;
    out[out_idx] = result;
}

void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out) {
    int batch = x.size(0);
    int in_channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);
    
    int out_channels = w.size(0);
    int kernel_h = w.size(2);
    int kernel_w = w.size(3);
    
    int out_height = out.size(2);
    int out_width = out.size(3);
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        batch
    );
    
    fused_conv_min_tanh_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        in_channels,
        out_channels,
        height,
        width,
        kernel_h,
        kernel_w,
        out_height,
        out_width
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward, "Fused Conv Min Tanh Forward");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
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
    # Calculate output dimensions (assuming padding=0, stride=1, dilation=1)
    batch_size = x.shape[0]
    out_height = x.shape[2] - conv_weight.shape[2] + 1
    out_width = x.shape[3] - conv_weight.shape[3] + 1
    out_channels = 1  # Because of min reduction over channels
    
    out = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    fused_ext.fused_op_forward(x, conv_weight, conv_bias, out)
    return out

batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
