# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_130922/code_19.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose3d_post_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ post_bias,
    float* __restrict__ output,
    int batch_size, int in_c, int out_c,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int k, int s, int p
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_c * out_d * out_h * out_w;
    if (idx >= total) return;

    int tmp = idx;
    int w_out = tmp % out_w; tmp /= out_w;
    int h_out = tmp % out_h; tmp /= out_h;
    int d_out = tmp % out_d; tmp /= out_d;
    int oc = tmp % out_c;
    int n = tmp / out_c;

    float acc = conv_bias[oc];

    for (int ic = 0; ic < in_c; ++ic) {
        for (int kd = 0; kd < k; ++kd) {
            int id = d_out + p - kd;
            if (id < 0 || id >= in_d * s || id % s != 0) continue;
            id /= s;
            for (int kh = 0; kh < k; ++kh) {
                int ih = h_out + p - kh;
                if (ih < 0 || ih >= in_h * s || ih % s != 0) continue;
                ih /= s;
                for (int kw = 0; kw < k; ++kw) {
                    int iw = w_out + p - kw;
                    if (iw < 0 || iw >= in_w * s || iw % s != 0) continue;
                    iw /= s;
                    
                    int in_idx = ((n * in_c + ic) * in_d + id) * (in_h * in_w) + ih * in_w + iw;
                    int wt_idx = ((ic * out_c + oc) * k + kd) * (k * k) + kh * k + kw;
                    acc += input[in_idx] * weight[wt_idx];
                }
            }
        }
    }

    float pb = post_bias[oc];
    output[idx] = ((acc + pb) + acc) * acc + acc;
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_kernel(const torch::Tensor& in, const torch::Tensor& wt, const torch::Tensor& cb, const torch::Tensor& pb, torch::Tensor& out, int k, int s, int p);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_post", &launch_kernel, "Fused kernel");
}
"""

cuda_source_full = cuda_kernel + r"""
void launch_kernel(const torch::Tensor& in, const torch::Tensor& wt, const torch::Tensor& cb, const torch::Tensor& pb, torch::Tensor& out, int k, int s, int p) {
    int b = in.size(0), ic = in.size(1), id = in.size(2), ih = in.size(3), iw = in.size(4);
    int oc = wt.size(0), od = out.size(2), oh = out.size(3), ow = out.size(4);
    int total = b * oc * od * oh * ow;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fused_conv_transpose3d_post_kernel<<<blocks, threads>>>(
        in.data_ptr<float>(), wt.data_ptr<float>(), cb.data_ptr<float>(), pb.data_ptr<float>(), out.data_ptr<float>(),
        b, ic, oc, id, ih, iw, od, oh, ow, k, s, p
    );
}
"""

fused_ext = load_inline(name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_source_full, extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, conv_transpose_dilation, bias):
    stride, padding, out_pad = conv_transpose_stride[0], conv_transpose_padding[0], conv_transpose_output_padding[0]
    out_d = (x.size(2) - 1) * stride - 2 * padding + 3 + out_pad
    out_s = [x.size(0), conv_transpose_weight.size(1), out_d, out_d, out_d]
    out = torch.empty(out_s, device=x.device, dtype=x.dtype)
    return fused_ext.fused_conv_transpose3d_post(x, conv_transpose_weight, conv_transpose_bias, bias.view(-1), out, 3, stride, padding)
