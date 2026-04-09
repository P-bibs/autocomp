# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_051902/code_1.py
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

# CUDA kernel for fused convolution + hardswish + relu operation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void fused_conv_hardswish_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N, int C, int H, int W,
    int OC, int K, int pad, int stride, int dilation
) {
    // Calculate output position
    int ow = threadIdx.x + blockIdx.x * blockDim.x;
    int oh = threadIdx.y + blockIdx.y * blockDim.y;
    int oc = blockIdx.z;
    
    if (ow >= W || oh >= H) return;
    
    float sum = 0.0f;
    
    // Perform convolution
    for (int ic = 0; ic < C; ++ic) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int ih = oh * stride - pad + kh * dilation;
                int iw = ow * stride - pad + kw * dilation;
                
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float val = x[((blockIdx.z % N) * C + ic) * H * W + ih * W + iw];
                    float wgt = weight[oc * C * K * K + ic * K * K + kh * K + kw];
                    sum += val * wgt;
                }
            }
        }
    }
    
    // Add bias
    sum += bias[oc];
    
    // Apply hardswish: x * relu6(x + 3) / 6
    float hardswish_result = sum * fminf(fmaxf(sum + 3.0f, 0.0f), 6.0f) / 6.0f;
    
    // Apply relu
    float final_result = fmaxf(hardswish_result, 0.0f);
    
    // Write output
    out[((blockIdx.z % N) * OC + oc) * H * W + oh * W + ow] = final_result;
}

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out,
    int N, int C, int H, int W,
    int OC, int K, int pad, int stride, int dilation
) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y, N * OC);
    
    fused_conv_hardswish_relu_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W, OC, K, pad, stride, dilation
    );
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out,
    int N, int C, int H, int W,
    int OC, int K, int pad, int stride, int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv + HardSwish + ReLU forward pass");
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
    # Validate assumptions (3x3 kernel, stride=1, padding=1, dilation=1, groups=1)
    assert conv_weight.shape[2] == 3 and conv_weight.shape[3] == 3, "Kernel size must be 3x3"
    assert conv_stride == 1, "Only stride=1 supported"
    assert conv_padding == 1, "Only padding=1 supported"
    assert conv_dilation == 1, "Only dilation=1 supported"
    assert conv_groups == 1, "Only groups=1 supported"
    
    # Create output tensor
    out = torch.empty((x.shape[0], conv_weight.shape[0], x.shape[2], x.shape[3]), 
                      device=x.device, dtype=x.dtype)
    
    # Call fused CUDA kernel
    fused_ext.fused_op(
        x.contiguous(), 
        conv_weight.contiguous(), 
        conv_bias.contiguous(), 
        out,
        x.shape[0],    # N
        x.shape[1],    # C
        x.shape[2],    # H
        x.shape[3],    # W
        conv_weight.shape[0],  # OC
        3,             # K (kernel size)
        conv_padding,  # pad
        conv_stride,   # stride
        conv_dilation  # dilation
    )
    
    return out

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
