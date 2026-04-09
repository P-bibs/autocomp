# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_084802/code_11.py
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

# The CUDA kernel performs the full convolution, channel-wise min reduction, 
# and double-tanh activation in one pass. This avoids global memory round-trips 
# for intermediate convolution outputs.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int ic, int ih, int iw,
    int oc, int kh, int kw,
    int pad, int stride,
    int oh, int ow) 
{
    int b = blockIdx.z;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch && h_out < oh && w_out < ow) {
        float min_val = 1e38f;

        // Iterate over output channels to find the minimum
        for (int c = 0; c < oc; ++c) {
            float sum = bias[c];
            for (int ky = 0; ky < kh; ++ky) {
                for (int kx = 0; kx < kw; ++kx) {
                    int h_in = h_out * stride + ky - pad;
                    int w_in = w_out * stride + kx - pad;
                    if (h_in >= 0 && h_in < ih && w_in >= 0 && w_in < iw) {
                        int weight_base = c * (ic * kh * kw);
                        int input_base = (b * ic * ih * iw) + (h_in * iw + w_in);
                        for (int i = 0; i < ic; ++i) {
                            sum += input[input_base + i * (ih * iw)] * weight[weight_base + i * (kh * kw) + ky * kw + kx];
                        }
                    }
                }
            }
            if (sum < min_val) min_val = sum;
        }
        // Apply tanh twice as per original logic
        output[b * (oh * ow) + h_out * ow + w_out] = tanhf(tanhf(min_val));
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                      int stride, int pad) {
    int batch = input.size(0); int ic = input.size(1);
    int ih = input.size(2); int iw = input.size(3);
    int oc = weight.size(0); int kh = weight.size(2); int kw = weight.size(3);
    int oh = output.size(2); int ow = output.size(3);

    dim3 threads(16, 16);
    dim3 blocks((ow + 15) / 16, (oh + 15) / 16, batch);

    fused_conv_min_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch, ic, ih, iw, oc, kh, kw, pad, stride, oh, ow
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int pad);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused conv-min-tanh kernel");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    # conv_dilation and groups are assumed to be 1/default based on the provided original code structure
    batch, _, h, w = x.shape
    out_c, _, kh, kw = conv_weight.shape
    out_h = (h + 2 * conv_padding - kh) // conv_stride + 1
    out_w = (w + 2 * conv_padding - kw) // conv_stride + 1
    
    output = torch.empty((batch, 1, out_h, out_w), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_weight, conv_bias, output, conv_stride, conv_padding)
    return output
