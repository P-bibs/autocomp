# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_045050/code_14.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel implementation
# Note: This is an implementation of a direct ConvTranspose3d fused with Bias, Add, and Hardswish.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__device__ __forceinline__ float hardswish_impl(float x) {
    return x * fmaxf(0.0f, fminf(1.0f, (x + 3.0f) / 6.0f));
}

__global__ void fused_conv_transpose3d_add_hardswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int k_d, int k_h, int k_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    if (tid >= total_elements) return;

    int tmp = tid;
    int w_idx = tmp % out_w; tmp /= out_w;
    int h_idx = tmp % out_h; tmp /= out_h;
    int d_idx = tmp % out_d; tmp /= out_d;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    float acc = (bias != nullptr) ? bias[oc] : 0.0f;

    // Direct implementation of ConvTranspose3d (Gradient definition)
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < k_d; ++kd) {
            for (int kh = 0; kh < k_h; ++kh) {
                for (int kw = 0; kw < k_w; ++kw) {
                    int id = (d_idx + pad_d - kd);
                    int ih = (h_idx + pad_h - kh);
                    int iw = (w_idx + pad_w - kw);

                    if (id % stride_d == 0 && ih % stride_h == 0 && iw % stride_w == 0) {
                        id /= stride_d; ih /= stride_h; iw /= stride_w;
                        if (id >= 0 && id < in_d && ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            float val = input[b * (in_channels * in_d * in_h * in_w) + 
                                            ic * (in_d * in_h * in_w) + 
                                            id * (in_h * in_w) + ih * in_w + iw];
                            float w = weight[oc * (in_channels * k_d * k_h * k_w) + 
                                           ic * (k_d * k_h * k_w) + 
                                           kd * (k_h * k_w) + kh * k_w + kw];
                            acc += val * w;
                        }
                    }
                }
            }
        }
    }

    acc += add_input[tid];
    output[tid] = hardswish_impl(acc) * acc;
}

void fused_op_forward(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor add_input,
    torch::Tensor output, int stride, int padding, int out_pad
) {
    const int b = x.size(0);
    const int ic = x.size(1);
    const int oc = weight.size(0);
    const int kd = weight.size(2), kh = weight.size(3), kw = weight.size(4);
    const int id = x.size(2), ih = x.size(3), iw = x.size(4);
    const int od = output.size(2), oh = output.size(3), ow = output.size(4);

    int total = b * oc * od * oh * ow;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_conv_transpose3d_add_hardswish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        add_input.data_ptr<float>(), output.data_ptr<float>(),
        b, ic, oc, id, ih, iw, od, oh, ow, kd, kh, kw, 
        stride, stride, stride, padding, padding, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused ConvTranspose3d Add Hardswish");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(
    x, add_input, *, conv_transpose_weight, conv_transpose_bias, 
    conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding,
    conv_transpose_groups, conv_transpose_dilation, bias
):
    # Constraint: Groups = 1 for this kernel implementation
    s, p, op = conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding
    out_shape = (x.shape[0], conv_transpose_weight.shape[0], 
                 (x.shape[2]-1)*s - 2*p + conv_transpose_weight.shape[2] + op,
                 (x.shape[3]-1)*s - 2*p + conv_transpose_weight.shape[3] + op,
                 (x.shape[4]-1)*s - 2*p + conv_transpose_weight.shape[4] + op)
    
    output = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, add_input, output, s, p, op)
    return output
