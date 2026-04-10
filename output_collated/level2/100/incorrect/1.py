# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_113742/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'min_value', 'divisor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'min_value', 'divisor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

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
    if 'min_value' in flat_state:
        state_kwargs['min_value'] = flat_state['min_value']
    else:
        state_kwargs['min_value'] = getattr(model, 'min_value')
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
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

# Optimization: Implement a full custom CUDA kernel for ConvTranspose3d fused with clamp and div
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__forceinline__ __device__ int get_linear_id_5d(int d0, int d1, int d2, int d3, int d4,
                                                int s0, int s1, int s2, int s3, int s4) {
    return ((((d0 * s0 + d1) * s1 + d2) * s2 + d3) * s3 + d4);
}

__global__ void fused_conv_transpose3d_clamp_div_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int K_D, int K_H, int K_W,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    float min_val, float divisor)
{
    int od = blockIdx.z;
    int oh = blockIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;

    if (od >= D_out || oh >= H_out || ow >= W_out) return;

    int kernel_size = K_D * K_H * K_W;
    int input_spatial_size = D_in * H_in * W_in;
    int output_spatial_size = D_out * H_out * W_out;

    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < C_out; ++oc) {
            float acc = (bias != nullptr) ? bias[oc] : 0.0f;

            for (int ic = 0; ic < C_in; ++ic) {
                for (int kd = 0; kd < K_D; ++kd) {
                    for (int kh = 0; kh < K_H; ++kh) {
                        for (int kw = 0; kw < K_W; ++kw) {
                            // Compute corresponding input location
                            int id = od + pad_d - kd * dilation_d;
                            int ih = oh + pad_h - kh * dilation_h;
                            int iw = ow + pad_w - kw * dilation_w;

                            if (id % stride_d == 0 && ih % stride_h == 0 && iw % stride_w == 0) {
                                id /= stride_d;
                                ih /= stride_h;
                                iw /= stride_w;

                                if (id >= 0 && id < D_in && ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                    int input_idx = get_linear_id_5d(n, ic, id, ih, iw, 
                                                                     C_in, D_in, H_in, W_in, 1);
                                    int weight_idx = get_linear_id_5d(oc, ic, kd, kh, kw,
                                                                      C_in, K_D, K_H, K_W, 1);
                                    acc += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                }
            }

            // Apply fused clamp and division
            acc = fmaxf(acc, min_val);
            acc = acc / divisor;

            int output_idx = get_linear_id_5d(n, oc, od, oh, ow,
                                              C_out, D_out, H_out, W_out, 1);
            output[output_idx] = acc;
        }
    }
}

void fused_conv_transpose3d_clamp_div_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    float min_val, float divisor)
{
    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    int C_out = weight.size(0);
    int K_D = weight.size(2);
    int K_H = weight.size(3);
    int K_W = weight.size(4);

    int D_out = (D_in - 1) * stride_d - 2 * padding_d + dilation_d * (K_D - 1) + output_padding_d + 1;
    int H_out = (H_in - 1) * stride_h - 2 * padding_h + dilation_h * (K_H - 1) + output_padding_h + 1;
    int W_out = (W_in - 1) * stride_w - 2 * padding_w + dilation_w * (K_W - 1) + output_padding_w + 1;

    dim3 block(256);
    dim3 grid((W_out + block.x - 1) / block.x, H_out, D_out);

    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

    fused_conv_transpose3d_clamp_div_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        K_D, K_H, K_W,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w,
        min_val, divisor
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
#include <optional>

void fused_conv_transpose3d_clamp_div_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    float min_val, float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_clamp_div_forward", &fused_conv_transpose3d_clamp_div_forward, "Fused ConvTranspose3d with clamp and div");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose3d_clamp_div',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    min_value,
    divisor,
):
    # Validate groups (only supporting groups=1 for this implementation)
    if conv_transpose_groups != 1:
        raise NotImplementedError("Only groups=1 is supported in this optimized implementation")
    
    # Create output tensor with correct shape
    N, C_in, D_in, H_in, W_in = x.shape
    C_out, _, K_D, K_H, K_W = conv_transpose_weight.shape
    
    # Calculate output dimensions
    stride_d, stride_h, stride_w = conv_transpose_stride
    pad_d, pad_h, pad_w = conv_transpose_padding
    out_pad_d, out_pad_h, out_pad_w = conv_transpose_output_padding
    dil_d, dil_h, dil_w = conv_transpose_dilation
    
    D_out = (D_in - 1) * stride_d - 2 * pad_d + dil_d * (K_D - 1) + out_pad_d + 1
    H_out = (H_in - 1) * stride_h - 2 * pad_h + dil_h * (K_H - 1) + out_pad_h + 1
    W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (K_W - 1) + out_pad_w + 1
    
    output = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Call our custom fused kernel
    fused_ext.fused_conv_transpose3d_clamp_div_forward(
        x, conv_transpose_weight, conv_transpose_bias, output,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w,
        dil_d, dil_h, dil_w,
        min_value, divisor
    )
    
    return output

batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 24, 48, 48
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
