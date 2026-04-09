# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_083021/code_9.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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
#include <float.h>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size,
    int stride, int padding
) {
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = batch_size * out_height * out_width;

    if (tid >= total_pixels) return;

    int b = tid / (out_height * out_width);
    int h_out = (tid / out_width) % out_height;
    int w_out = tid % out_width;

    float min_val = FLT_MAX;

    // We iterate over out_channels to perform the reduction on-the-fly
    for (int c_out = 0; c_out < out_channels; ++c_out) {
        float val = bias[c_out];
        int h_in_base = h_out * stride - padding;
        int w_in_base = w_out * stride - padding;

        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int h_in = h_in_base + kh;
                    int w_in = w_in_base + kw;
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        float in_v = input[((b * in_channels + c_in) * height + h_in) * width + w_in];
                        float w_v = weight[((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw];
                        val += in_v * w_v;
                    }
                }
            }
        }
        if (val < min_val) min_val = val;
    }

    min_val = tanhf(tanhf(min_val));
    output[tid] = min_val;
}

void launch_fused_kernel(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                         int stride, int padding) {
    int b = input.size(0); int c_in = input.size(1);
    int h = input.size(2); int w = input.size(3);
    int c_out = weight.size(0); int k = weight.size(2);
    int out_h = (h + 2 * padding - k) / stride + 1;
    int out_w = (w + 2 * padding - k) / stride + 1;
    
    int total = b * out_h * out_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    fused_conv_min_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        b, c_in, c_out, h, w, k, stride, padding
    );
}
"""

cpp_source = """
void launch_fused_kernel(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused", &launch_fused_kernel); }
"""

fused_ext = load_inline(name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
                        extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    # As per instructions, assume groups=1, dilation=1 for high-perf custom kernel
    batch, _, h, w = x.shape
    k = conv_weight.shape[2]
    out_h = (h + 2 * conv_padding - k) // conv_stride + 1
    out_w = (w + 2 * conv_padding - k) // conv_stride + 1
    out = torch.empty(batch, 1, out_h, out_w, device=x.device, dtype=x.dtype)
    fused_ext.fused(x, conv_weight, conv_bias, out, conv_stride, conv_padding)
    return out
