# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_121229/code_19.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int Ci, int Co, int Hi, int Wi, int Ho, int Wo,
    int k, int stride, int padding, int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * Co * Ho * Wo;
    if (idx >= total_elements) return;

    int tmp = idx;
    int ow = tmp % Wo; tmp /= Wo;
    int oh = tmp % Ho; tmp /= Ho;
    int co = tmp % Co; tmp /= Co;
    int b = tmp;

    int group_size = Co / groups;
    int group_idx = co / group_size;
    int ci_start = group_idx * (Ci / groups);
    int ci_end = ci_start + (Ci / groups);

    float val = (conv_bias != nullptr) ? conv_bias[co] : 0.0f;

    for (int ci = ci_start; ci < ci_end; ++ci) {
        for (int kh = 0; kh < k; ++kh) {
            for (int kw = 0; kw < k; ++kw) {
                int ih = (oh + padding - kh) / stride;
                int iw = (ow + padding - kw) / stride;
                if (ih >= 0 && ih < Hi && iw >= 0 && iw < Wi && (oh + padding - kh) % stride == 0 && (ow + padding - kw) % stride == 0) {
                    int w_idx = (group_idx * (Ci / groups) * Co / groups + (ci - ci_start) * Co / groups + co % group_size) * k * k + kh * k + kw;
                    int i_idx = ((b * Ci + ci) * Hi + ih) * Wi + iw;
                    val += input[i_idx] * weight[w_idx];
                }
            }
        }
    }
    output[idx] = tanhf(val - bias[co]);
}

void fused_op_forward(
    const torch::Tensor& x, const torch::Tensor& weight, 
    const torch::Tensor& conv_bias, const torch::Tensor& bias, 
    torch::Tensor& out, int stride, int padding, int groups
) {
    int B = x.size(0), Ci = x.size(1), Hi = x.size(2), Wi = x.size(3);
    int Co = weight.size(1), k = weight.size(2);
    int Ho = out.size(2), Wo = out.size(3);
    
    int threads = 256;
    int blocks = (B * Co * Ho * Wo + threads - 1) / threads;
    
    fused_conv_transpose2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), 
        conv_bias.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), B, Ci, Co, Hi, Wi, Ho, Wo, k, stride, padding, groups
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& cb, const torch::Tensor& b, torch::Tensor& out, int s, int p, int g);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward);
}
"""

fused_ext = load_inline(
    name='fused_conv_tanh', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    B, Ci, Hi, Wi = x.shape
    _, Co, k, _ = conv_transpose_weight.shape
    Ho = (Hi - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (k - 1) + conv_transpose_output_padding + 1
    Wo = (Wi - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (k - 1) + conv_transpose_output_padding + 1
    
    out = torch.empty((B, Co, Ho, Wo), device=x.device)
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, bias.squeeze(), out, 
                       conv_transpose_stride, conv_transpose_padding, conv_transpose_groups)
    return out
