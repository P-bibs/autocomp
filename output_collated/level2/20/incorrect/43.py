# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_132207/code_22.py
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

# We implement an Implicit GEMM-based Transposed Convolution fused with the post-processing
# Arithmetic: out = ((val + bias) + val) * val + val
# This implementation focuses on accuracy and kernel fusion for the provided constraints.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int IC, int D, int H, int W,
    int OC, int KD, int KH, int KW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_D, int out_H, int out_W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = N * OC * out_D * out_H * out_W;
    
    if (idx >= total_output_elements) return;

    // Coordinate calculation
    int tmp = idx;
    int ow = tmp % out_W; tmp /= out_W;
    int oh = tmp % out_H; tmp /= out_H;
    int od = tmp % out_D; tmp /= out_D;
    int oc = tmp % OC; tmp /= OC;
    int n = tmp;

    float acc = 0.0f;

    // Implicit GEMM logic: Iterate over input patches that contribute to this output pixel
    for (int ic = 0; ic < IC; ++ic) {
        for (int kd = 0; kd < KD; ++kd) {
            int id = (od + pad_d - kd);
            if (id % stride_d != 0) continue;
            id /= stride_d;
            if (id < 0 || id >= D) continue;

            for (int kh = 0; kh < KH; ++kh) {
                int ih = (oh + pad_h - kh);
                if (ih % stride_h != 0) continue;
                ih /= stride_h;
                if (ih < 0 || ih >= H) continue;

                for (int kw = 0; kw < KW; ++kw) {
                    int iw = (ow + pad_w - kw);
                    if (iw % stride_w != 0) continue;
                    iw /= stride_w;
                    if (iw < 0 || iw >= W) continue;

                    float i_val = input[(((n * IC + ic) * D + id) * H + ih) * W + iw];
                    float w_val = weight[(((ic * OC + oc) * KD + kd) * KH + kh) * KW + kw];
                    acc += i_val * w_val;
                }
            }
        }
    }

    // Fused post-processing arithmetic
    float b = bias[oc];
    float res = ((acc + b) + acc) * acc + acc;
    output[idx] = res;
}

void launch_fused_conv(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w
) {
    int N = input.size(0);
    int IC = input.size(1);
    int D = input.size(2); int H = input.size(3); int W = input.size(4);
    int OC = weight.size(1);
    int KD = weight.size(2); int KH = weight.size(3); int KW = weight.size(4);
    int out_D = output.size(2); int out_H = output.size(3); int out_W = output.size(4);

    int total_elements = N * OC * out_D * out_H * out_W;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, IC, D, H, W, OC, KD, KH, KW,
        stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, out_D, out_H, out_W
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_fused_conv(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int s_d, int s_h, int s_w, int p_d, int p_h, int p_w);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &launch_fused_conv, "Fused Transposed Conv3D");
}
"""

fused_ext = load_inline(
    name='fused_conv_impl',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride,
                    conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups,
                    conv_transpose_dilation, bias):
    # Output shape calculation
    out_d = (x.shape[2]-1)*conv_transpose_stride[0] - 2*conv_transpose_padding[0] + (conv_transpose_weight.shape[2]-1) + conv_transpose_output_padding[0] + 1
    out_h = (x.shape[3]-1)*conv_transpose_stride[1] - 2*conv_transpose_padding[1] + (conv_transpose_weight.shape[3]-1) + conv_transpose_output_padding[1] + 1
    out_w = (x.shape[4]-1)*conv_transpose_stride[2] - 2*conv_transpose_padding[2] + (conv_transpose_weight.shape[4]-1) + conv_transpose_output_padding[2] + 1
    
    output = torch.empty((x.shape[0], conv_transpose_weight.shape[1], out_d, out_h, out_w), device=x.device)
    
    fused_ext.fused_conv(
        x.contiguous(), conv_transpose_weight.contiguous(), bias.view(-1).contiguous(), output,
        conv_transpose_stride[0], conv_transpose_stride[1], conv_transpose_stride[2],
        conv_transpose_padding[0], conv_transpose_padding[1], conv_transpose_padding[2]
    )
    return output
