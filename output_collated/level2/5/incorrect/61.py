# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_121229/code_27.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'bias_shape', 'stride', 'padding', 'output_padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
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
    # State for conv_transpose (nn.ConvTranspose2d)
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

# CUDA kernel for fused conv_transpose2d + subtraction + tanh
# This kernel implements the transposed convolution operation directly 
# to avoid the overhead of calling PyTorch's native functions.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose_sub_tanh_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ sub_bias,
    float* __restrict__ output,
    int B, int IC, int OC, int IH, int IW,
    int KH, int KW, int OH, int OW,
    int stride, int padding, int dilation, int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * OC * OH * OW;
    if (idx >= total_elements) return;

    int tmp = idx;
    int w = tmp % OW; tmp /= OW;
    int h = tmp % OH; tmp /= OH;
    int oc = tmp % OC; tmp /= OC;
    int b = tmp;

    int group_id = oc / (OC / groups);
    int ic_start = group_id * (IC / groups);
    int ic_end = ic_start + (IC / groups);

    float val = (conv_bias != nullptr) ? conv_bias[oc] : 0.0f;

    for (int ic = ic_start; ic < ic_end; ++ic) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                // In gradient/transpose conv: output coord = input_coord * stride + kh - padding
                // Here we reverse map: input_coord = (output_coord - kh + padding) / stride
                int ih_raw = h + padding - kh * dilation;
                int iw_raw = w + padding - kw * dilation;

                if (ih_raw % stride == 0 && iw_raw % stride == 0) {
                    int ih = ih_raw / stride;
                    int iw = iw_raw / stride;

                    if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                        int x_idx = ((b * IC + ic) * IH + ih) * IW + iw;
                        int w_idx = ((oc * (IC / groups) + (ic - ic_start)) * KH + kh) * KW + kw;
                        val += x[x_idx] * weight[w_idx];
                    }
                }
            }
        }
    }

    float sub = (sub_bias != nullptr) ? sub_bias[oc] : 0.0f;
    output[idx] = tanhf(val - sub);
}

void fused_op_forward(
    torch::Tensor x, torch::Tensor weight, torch::Tensor conv_bias,
    torch::Tensor sub_bias, torch::Tensor output,
    int stride, int padding, int output_padding, int groups, int dilation
) {
    int B = x.size(0); int IC = x.size(1);
    int OC = weight.size(1) * groups;
    int IH = x.size(2); int IW = x.size(3);
    int KH = weight.size(2); int KW = weight.size(3);
    int OH = output.size(2); int OW = output.size(3);

    int total_elements = B * OC * OH * OW;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_conv_transpose_sub_tanh_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(),
        conv_bias.defined() ? conv_bias.data_ptr<float>() : nullptr,
        sub_bias.defined() ? sub_bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B, IC, OC, IH, IW, KH, KW, OH, OW,
        stride, padding, dilation, groups
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor conv_bias,
                      torch::Tensor sub_bias, torch::Tensor output,
                      int stride, int padding, int output_padding, int groups, int dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused conv_transpose2d + sub + tanh");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(
    x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride,
    conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups,
    conv_transpose_dilation, bias
):
    B, IC, IH, IW = x.shape
    OC, _, KH, KW = conv_transpose_weight.shape
    OC *= conv_transpose_groups
    
    OH = (IH - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (KH - 1) + conv_transpose_output_padding + 1
    OW = (IW - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (KW - 1) + conv_transpose_output_padding + 1
    
    output = torch.empty((B, OC, OH, OW), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(
        x, conv_transpose_weight, conv_transpose_bias, bias, output,
        conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding,
        conv_transpose_groups, conv_transpose_dilation
    )
    return output
