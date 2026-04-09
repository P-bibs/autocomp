# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_115905/code_14.py
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

# CUDA kernel implementation
# Note: For production-grade performance, libraries like cuDNN would be used, 
# but per the prompt, we implement the operator directly in CUDA.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ sub_bias,
    float* __restrict__ output,
    int batch, int ic, int ih, int iw,
    int oc, int kh, int kw,
    int stride, int pad,
    int oh, int ow
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * oc * oh * ow;
    if (tid >= total) return;

    int w_out = tid % ow;
    int h_out = (tid / ow) % oh;
    int c_out = (tid / (ow * oh)) % oc;
    int b = tid / (ow * oh * oc);

    float val = conv_bias[c_out];

    // Transposed Convolution iteration
    for (int ic_idx = 0; ic_idx < ic; ++ic_idx) {
        for (int i = 0; i < kh; ++i) {
            for (int j = 0; j < kw; ++j) {
                int h_in = h_out - i + pad;
                int w_in = w_out - j + pad;
                if (h_in % stride == 0 && w_in % stride == 0) {
                    h_in /= stride;
                    w_in /= stride;
                    if (h_in >= 0 && h_in < ih && w_in >= 0 && w_in < iw) {
                        float in_val = input[((b * ic + ic_idx) * ih + h_in) * iw + w_in];
                        float w_val = weight[((c_out * ic + ic_idx) * kh + i) * kw + j];
                        val += in_val * w_val;
                    }
                }
            }
        }
    }
    
    val -= sub_bias[c_out];
    output[tid] = tanhf(val);
}

void fused_op_forward(
    const torch::Tensor input, const torch::Tensor weight,
    const torch::Tensor conv_bias, const torch::Tensor sub_bias,
    torch::Tensor output, int stride, int pad, int kh, int kw
) {
    int batch = input.size(0);
    int ic = input.size(1);
    int ih = input.size(2);
    int iw = input.size(3);
    int oc = output.size(1);
    int oh = output.size(2);
    int ow = output.size(3);

    int total = batch * oc * oh * ow;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_conv_transpose_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(), sub_bias.data_ptr<float>(),
        output.data_ptr<float>(), batch, ic, ih, iw, oc, kh, kw, stride, pad, oh, ow
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(const torch::Tensor input, const torch::Tensor weight, const torch::Tensor conv_bias, const torch::Tensor sub_bias, torch::Tensor output, int stride, int pad, int kh, int kw);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Transposed Conv + Bias + Tanh");
}
"""

fused_ext = load_inline(
    name='fused_conv_tanh', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    stride = conv_transpose_stride[0]
    pad = conv_transpose_padding[0]
    kh = conv_transpose_weight.shape[2]
    kw = conv_transpose_weight.shape[3]
    oh = (x.shape[2] - 1) * stride - 2 * pad + kh + conv_transpose_output_padding[0]
    ow = (x.shape[3] - 1) * stride - 2 * pad + kw + conv_transpose_output_padding[1]
    
    output = torch.empty((x.shape[0], conv_transpose_weight.shape[1], oh, ow), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, bias.view(-1), output, stride, pad, kh, kw)
    return output
