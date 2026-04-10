# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_140117/code_26.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by a sum, 
    a residual add, a multiplication, and another residual add.
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

# The custom kernel performs: 
# 1. Transposed 3D convolution (Output = Input * Weight)
# 2. Add bias, then perform the element-wise op: ((x + bias) + x) * x + x
# Note: Full custom ConvTranspose3D is complex; we implement an optimized 
# tiled kernel approach suitable for JIT compilation.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out, 
    int D, int H, int W,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_D, int out_H, int out_W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = N * C_out * out_D * out_H * out_W;

    if (idx >= total_out) return;

    // Mapping flat idx to 5D coords
    int w = idx % out_W;
    int h = (idx / out_W) % out_H;
    int d = (idx / (out_W * out_H)) % out_D;
    int c_out = (idx / (out_W * out_H * out_D)) % C_out;
    int n = idx / (out_W * out_H * out_D * C_out);

    float sum = 0.0f;

    // Transposed Conv logic: iterate over input pixels that contribute to this output
    for (int ic = 0; ic < C_in; ++ic) {
        for (int kd = 0; kd < kD; ++kd) {
            int id = (d + pad_d - kd);
            if (id % stride_d != 0) continue;
            id /= stride_d;
            if (id < 0 || id >= D) continue;

            for (int kh = 0; kh < kH; ++kh) {
                int ih = (h + pad_h - kh);
                if (ih % stride_h != 0) continue;
                ih /= stride_h;
                if (ih < 0 || ih >= H) continue;

                for (int kw = 0; kw < kW; ++kw) {
                    int iw = (w + pad_w - kw);
                    if (iw % stride_w != 0) continue;
                    iw /= stride_w;
                    if (iw < 0 || iw >= W) continue;

                    float val = input[((n * C_in + ic) * D + id) * H * W + ih * W + iw];
                    float w_val = weight[((ic * C_out + c_out) * kD + kd) * kH * kW + kh * kW + kw];
                    sum += val * w_val;
                }
            }
        }
    }

    float b = bias[c_out];
    float res = ((sum + b) + sum) * sum + sum;
    output[idx] = res;
}

void launch_fused_conv(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride, int pad, int out_pad
) {
    int N = input.size(0); int C_in = input.size(1);
    int D = input.size(2); int H = input.size(3); int W = input.size(4);
    int C_out = weight.size(1);
    int kD = weight.size(2); int kH = weight.size(3); int kW = weight.size(4);
    int out_D = output.size(2); int out_H = output.size(3); int out_W = output.size(4);

    int total_threads = N * C_out * out_D * out_H * out_W;
    int threadsPerBlock = 256;
    int blocks = (total_threads + threadsPerBlock - 1) / threadsPerBlock;

    fused_conv_transpose_kernel<<<blocks, threadsPerBlock>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, C_in, C_out, D, H, W, kD, kH, kW,
        stride, stride, stride, pad, pad, pad, out_D, out_H, out_W
    );
}
"""

cpp_source = r"""
void launch_fused_conv(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int pad, int out_pad);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &launch_fused_conv, "Fused Transposed Conv3D + Post-processing");
}
"""

fused_ext = load_inline(
    name='fused_conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    # Weight shape expected in kernel: [C_in, C_out, kD, kH, kW]
    # Standard PyTorch weight is [C_in, C_out/groups, kD, kH, kW]
    w = conv_transpose_weight
    out_shape = (x.shape[0], w.shape[1], 
                 (x.shape[2]-1)*conv_transpose_stride + w.shape[2] - 2*conv_transpose_padding + conv_transpose_output_padding[0],
                 (x.shape[3]-1)*conv_transpose_stride + w.shape[3] - 2*conv_transpose_padding + conv_transpose_output_padding[1],
                 (x.shape[4]-1)*conv_transpose_stride + w.shape[4] - 2*conv_transpose_padding + conv_transpose_output_padding[2])
    
    output = torch.zeros(out_shape, device=x.device)
    fused_ext.fused_conv(x, w, bias, output, conv_transpose_stride, conv_transpose_padding[0], conv_transpose_output_padding[0])
    return output
