# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_043234/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x', 'add_input']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

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
    # State for conv_transpose (nn.ConvTranspose3d)
    if 'conv_transpose_weight' in flat_state:
        state_kwargs['conv_transpose_weight'] = flat_state['conv_transpose_weight']
    else:
        state_kwargs['conv_transpose_weight'] = getattr(model.conv_transpose, 'weight', None)
    if 'conv_transpose_bias' in flat_state:
        state_kwargs['conv_transpose_bias'] = flat_state['conv_transpose_bias']
    else:
        state_kwargs['conv_transpose_bias'] = getattr(model.conv_transpose, 'bias', None)
    state_kwargs['conv_transpose_stride'] = model.conv_transpose.stride
    state_kwargs['conv_transpose_padding'] = model.conv_transpose.padding
    state_kwargs['conv_transpose_output_padding'] = model.conv_transpose.output_padding
    state_kwargs['conv_transpose_groups'] = model.conv_transpose.groups
    state_kwargs['conv_transpose_dilation'] = model.conv_transpose.dilation
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
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

# CUDA kernel for fused operation: conv_transpose3d + add + hardswish
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float hardswish_impl(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

__global__ void fused_conv_transpose3d_add_hardswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ add_input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int Ci, int Co,
    int D, int H, int W,
    int KD, int KH, int KW,
    int stride, int padding, int output_padding,
    int dilation, int groups
) {
    int OD = (D - 1) * stride - 2 * padding + dilation * (KD - 1) + output_padding + 1;
    int OH = (H - 1) * stride - 2 * padding + dilation * (KH - 1) + output_padding + 1;
    int OW = (W - 1) * stride - 2 * padding + dilation * (KW - 1) + output_padding + 1;

    int total_threads = B * Co * OD * OH * OW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_threads) return;

    int ow = idx % OW; idx /= OW;
    int oh = idx % OH; idx /= OH;
    int od = idx % OD; idx /= OD;
    int co = idx % Co; idx /= Co;
    int b = idx;

    float acc = 0.0f;
    int group = co / (Co / groups);
    
    for (int ci = group * (Ci / groups); ci < (group + 1) * (Ci / groups); ++ci) {
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int d = od + padding - kd * dilation;
                    int h = oh + padding - kh * dilation;
                    int w = ow + padding - kw * dilation;
                    
                    if (d % stride == 0 && h % stride == 0 && w % stride == 0) {
                        d /= stride;
                        h /= stride;
                        w /= stride;
                        
                        if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) {
                            int input_idx = b * (Ci * D * H * W) + ci * (D * H * W) + d * (H * W) + h * W + w;
                            int weight_idx = co * (Ci / groups * KD * KH * KW) + (ci - group * (Ci / groups)) * (KD * KH * KW) + (KD - 1 - kd) * (KH * KW) + (KH - 1 - kh) * KW + (KW - 1 - kw);
                            acc += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }

    acc += bias[co];
    acc += add_input[b * (Co * OD * OH * OW) + co * (OD * OH * OW) + od * (OH * OW) + oh * OW + ow];
    output[b * (Co * OD * OH * OW) + co * (OD * OH * OW) + od * (OH * OW) + oh * OW + ow] = acc * hardswish_impl(acc);
}

void launch_fused_op(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& add_input,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int KD, int KH, int KW,
    int stride, int padding, int output_padding,
    int dilation, int groups
) {
    int B = input.size(0);
    int Ci = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    int Co = weight.size(0);
    
    int OD = (D - 1) * stride - 2 * padding + dilation * (KD - 1) + output_padding + 1;
    int OH = (H - 1) * stride - 2 * padding + dilation * (KH - 1) + output_padding + 1;
    int OW = (W - 1) * stride - 2 * padding + dilation * (KW - 1) + output_padding + 1;

    int total_elements = B * Co * OD * OH * OW;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_conv_transpose3d_add_hardswish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        add_input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, Ci, Co, D, H, W, KD, KH, KW,
        stride, padding, output_padding, dilation, groups
    );
}
"""

# C++ binding
cpp_source = r"""
#include <torch/extension.h>

void launch_fused_op(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& add_input,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int KD, int KH, int KW,
    int stride, int padding, int output_padding,
    int dilation, int groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_op", &launch_fused_op, "Fused ConvTranspose3d + Add + Hardswish");
}
"""

# Load the extension
fused_ext = load_inline(
    name='fused_conv_transpose3d_add_hardswish',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    add_input,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    bias=None,
):
    # Kernel dimensions from weight shape (Co, Ci/groups, KD, KH, KW)
    KD, KH, KW = conv_transpose_weight.shape[-3:]
    
    # Create output tensor with correct shape
    B, Ci, D, H, W = x.shape
    Co = conv_transpose_weight.shape[0]
    stride = conv_transpose_stride
    padding = conv_transpose_padding
    output_padding = conv_transpose_output_padding
    dilation = conv_transpose_dilation
    groups = conv_transpose_groups
    
    OD = (D - 1) * stride - 2 * padding + dilation * (KD - 1) + output_padding + 1
    OH = (H - 1) * stride - 2 * padding + dilation * (KH - 1) + output_padding + 1
    OW = (W - 1) * stride - 2 * padding + dilation * (KW - 1) + output_padding + 1
    
    output = torch.empty((B, Co, OD, OH, OW), dtype=x.dtype, device=x.device)
    
    # Launch the fused kernel
    fused_ext.launch_fused_op(
        x, conv_transpose_weight, add_input, conv_transpose_bias,
        output, KD, KH, KW, stride, padding, output_padding, dilation, groups
    )
    
    return output

# Constants
batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W), torch.rand(batch_size, out_channels, D*stride, H*stride, W*stride)]
