# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_132110/code_4.py
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

# CUDA kernel that computes fused ConvTranspose3d + Clamp + Div
# We use a direct implementation of the transpose convolution logic
# as required by the constraint to avoid built-in torch convolution functions.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_d, const int in_h, const int in_w,
    const int out_d, const int out_h, const int out_w,
    const int k, const int stride, const int pad,
    const float min_val, const float div) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    if (tid >= total_elements) return;

    int w = tid % out_w;
    int h = (tid / out_w) % out_h;
    int d = (tid / (out_w * out_h)) % out_d;
    int c = (tid / (out_w * out_h * out_d)) % out_channels;
    int b = tid / (out_w * out_h * out_d * out_channels);

    float val = bias[c];

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < k; ++kd) {
            for (int kh = 0; kh < k; ++kh) {
                for (int kw = 0; kw < k; ++kw) {
                    int id = (d + pad - kd);
                    int ih = (h + pad - kh);
                    int iw = (w + pad - kw);

                    if (id % stride == 0 && ih % stride == 0 && iw % stride == 0) {
                        id /= stride;
                        ih /= stride;
                        iw /= stride;

                        if (id >= 0 && id < in_d && ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            int in_idx = b * (in_channels * in_d * in_h * in_w) +
                                         ic * (in_d * in_h * in_w) +
                                         id * (in_h * in_w) +
                                         ih * in_w +
                                         iw;
                            int w_idx = c * (in_channels * k * k * k) +
                                        ic * (k * k * k) +
                                        kd * (k * k) +
                                        kh * k +
                                        kw;
                            val += input[in_idx] * weight[w_idx];
                        }
                    }
                }
            }
        }
    }

    val = fmaxf(min_val, val);
    output[tid] = val / div;
}

void launch_fused_op(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
    torch::Tensor& output, int batch_size, int in_channels, int out_channels,
    int in_d, int in_h, int in_w, int out_d, int out_h, int out_w,
    int k, int stride, int pad, float min_val, float div) {
    
    int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w, k, stride, pad, min_val, div
    );
}
"""

cpp_source = r"""
void launch_fused_op(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& output, int batch_size, int in_channels, int out_channels, int in_d, int in_h, int in_w, int out_d, int out_h, int out_w, int k, int stride, int pad, float min_val, float div);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused_op, "Fused ConvTranspose3d + Clamp + Div");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride,
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups,
                     conv_transpose_dilation, min_value, divisor):
    b, in_c, id, ih, iw = x.shape
    out_c = conv_transpose_weight.shape[1] 
    k = conv_transpose_weight.shape[2]
    stride = conv_transpose_stride[0]
    pad = conv_transpose_padding[0]
    
    out_d = (id - 1) * stride - 2 * pad + k
    out_h = (ih - 1) * stride - 2 * pad + k
    out_w = (iw - 1) * stride - 2 * pad + k
    
    output = torch.empty((b, out_c, out_d, out_h, out_w), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, output,
                       b, in_c, out_c, id, ih, iw, out_d, out_h, out_w,
                       k, stride, pad, float(min_value), float(divisor))
    return output
