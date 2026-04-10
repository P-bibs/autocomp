# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141921/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'subtract_value_1', 'subtract_value_2']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'subtract_value_1', 'subtract_value_2']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, subtracts two values, applies Mish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

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
    # State for conv (nn.Conv2d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
    if 'subtract_value_1' in flat_state:
        state_kwargs['subtract_value_1'] = flat_state['subtract_value_1']
    else:
        state_kwargs['subtract_value_1'] = getattr(model, 'subtract_value_1')
    if 'subtract_value_2' in flat_state:
        state_kwargs['subtract_value_2'] = flat_state['subtract_value_2']
    else:
        state_kwargs['subtract_value_2'] = getattr(model, 'subtract_value_2')
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

# Optimization: Merge high-level operations (Convolution + Subtractions + Mish)
# into a single custom CUDA kernel to minimize global memory round-trips.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int in_c, int in_h, int in_w,
    int out_c, int k_h, int k_w, int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    float sub1, float sub2) {

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch * out_c * out_h * out_w) return;

    int tmp = out_idx;
    int w_out = tmp % out_w; tmp /= out_w;
    int h_out = tmp % out_h; tmp /= out_h;
    int c_out = tmp % out_c; tmp /= out_c;
    int b = tmp;

    float acc = bias[c_out];
    
    for (int ic = 0; ic < in_c; ++ic) {
        for (int kh = 0; kh < k_h; ++kh) {
            for (int kw = 0; kw < k_w; ++kw) {
                int h_in = h_out * stride_h - pad_h + kh * dilation_h;
                int w_in = w_out * stride_w - pad_w + kw * dilation_w;
                
                if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                    float val = input[((b * in_c + ic) * in_h + h_in) * in_w + w_in];
                    float wgt = weight[(((c_out * in_c + ic) * k_h + kh) * k_w + kw)];
                    acc += val * wgt;
                }
            }
        }
    }

    // Fused operations: Subtractions + Mish
    float val = acc - sub1 - sub2;
    
    // Mish: x * tanh(softplus(x))
    float sp = (val > 20.0f) ? val : logf(1.0f + expf(val));
    output[out_idx] = val * tanhf(sp);
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor output, int stride_h, int stride_w,
                      int pad_h, int pad_w, int dilation_h, int dilation_w,
                      float sub1, float sub2) {
    int batch = input.size(0);
    int in_c = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_c = weight.size(0);
    int k_h = weight.size(2);
    int k_w = weight.size(3);
    int out_h = (in_h + 2 * pad_h - dilation_h * (k_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - dilation_w * (k_w - 1) - 1) / stride_w + 1;

    int total_elements = batch * out_c * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_conv_mish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch, in_c, in_h, in_w, out_c, k_h, k_w, 
        out_h, out_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, sub1, sub2);
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor output, int stride_h, int stride_w,
                      int pad_h, int pad_w, int dilation_h, int dilation_w,
                      float sub1, float sub2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv2d + Subtracts + Mish");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, 
                     conv_groups, subtract_value_1, subtract_value_2):
    # Handle different input formats for stride, padding, dilation
    if isinstance(conv_stride, int):
        stride_h, stride_w = conv_stride, conv_stride
    else:
        stride_h, stride_w = conv_stride[0], conv_stride[1]
        
    if isinstance(conv_padding, int):
        pad_h, pad_w = conv_padding, conv_padding
    else:
        pad_h, pad_w = conv_padding[0], conv_padding[1]
        
    if isinstance(conv_dilation, int):
        dilation_h, dilation_w = conv_dilation, conv_dilation
    else:
        dilation_h, dilation_w = conv_dilation[0], conv_dilation[1]
    
    out_h = (x.size(2) + 2 * pad_h - dilation_h * (conv_weight.size(2) - 1) - 1) // stride_h + 1
    out_w = (x.size(3) + 2 * pad_w - dilation_w * (conv_weight.size(3) - 1) - 1) // stride_w + 1
    
    output = torch.empty((x.size(0), conv_weight.size(0), out_h, out_w), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_weight, conv_bias, output, 
                       stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
                       subtract_value_1, subtract_value_2)
    return output

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
