# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_050905/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies HardSwish, and then ReLU.
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

# Optimized CUDA kernel implementing fused 3x3 convolution with hardswish and relu activations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv2d_act_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C, int H, int W, int OC
) {
    // Calculate global thread indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;
    int n = oc / OC;
    oc = oc % OC;

    if (x < W && y < H && oc < OC) {
        float val = bias[oc];
        
        // Perform 3x3 convolution (assuming stride=1, padding=1)
        for (int ic = 0; ic < C; ++ic) {
            for (int ky = 0; ky < 3; ++ky) {
                for (int kx = 0; kx < 3; ++kx) {
                    int iy = y + ky - 1;
                    int ix = x + kx - 1;
                    
                    // Handle padding (implicit zero padding)
                    if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                        val += input[((n * C + ic) * H + iy) * W + ix] *
                               weight[(((oc * C + ic) * 3 + ky) * 3 + kx)];
                    }
                }
            }
        }
        
        // Apply hardswish: x * relu6(x + 3) / 6
        float hswish = val * fminf(fmaxf(val + 3.0f, 0.0f), 6.0f) / 6.0f;
        
        // Apply relu
        float result = fmaxf(0.0f, hswish);
        
        output[((n * OC + oc) * H + y) * W + x] = result;
    }
}

void fused_op_forward(int B, int C, int H, int W, int OC, 
                      torch::Tensor input, torch::Tensor weight, 
                      torch::Tensor bias, torch::Tensor output) {
    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, 
              (H + block.y - 1) / block.y, 
              B * OC);
    
    fused_conv2d_act_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W, OC
    );
}
"""

# C++ bindings for the CUDA kernel
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(int B, int C, int H, int W, int OC, 
                      torch::Tensor input, torch::Tensor weight, 
                      torch::Tensor bias, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv2D + Hardswish + ReLU forward pass");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_act',
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
    # Validate assumptions for this optimized implementation
    assert conv_stride == 1, "Only stride=1 is supported in this implementation"
    assert conv_padding == 1, "Only padding=1 is supported in this implementation"
    assert conv_dilation == 1, "Only dilation=1 is supported in this implementation"
    assert conv_groups == 1, "Only groups=1 is supported in this implementation"
    assert conv_weight.size(2) == 3 and conv_weight.size(3) == 3, "Only 3x3 kernels are supported"
    
    # Create output tensor with the correct dimensions
    B, C, H, W = x.shape
    OC = conv_weight.shape[0]
    output = torch.empty((B, OC, H, W), device=x.device, dtype=x.dtype)
    
    # Call the fused CUDA kernel
    fused_ext.fused_op(B, C, H, W, OC, x, conv_weight, conv_bias, output)
    
    return output

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
