# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_051539/code_4.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies HardSwish, and then ReLU.
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

# The custom CUDA implementation of a 2D convolution fused with Hardswish and ReLU.
# This implementation performs the sliding window sum directly in a custom kernel,
# avoiding standard library calls.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_activation_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size,
    int stride, int padding, int dilation, int groups
) {
    int b = blockIdx.z / out_channels;
    int oc = blockIdx.z % out_channels;
    int oh = blockIdx.y;
    int ow = blockIdx.x;

    int out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    float acc = 0.0f;
    int ic_per_group = in_channels / groups;
    int group_id = oc / (out_channels / groups);
    int ic_start = group_id * ic_per_group;

    for (int ic = 0; ic < ic_per_group; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = oh * stride - padding + kh * dilation;
                int iw = ow * stride - padding + kw * dilation;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int in_idx = ((b * in_channels + (ic_start + ic)) * height + ih) * width + iw;
                    int w_idx = (((oc * ic_per_group) + ic) * kernel_size + kh) * kernel_size + kw;
                    acc += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    float val = acc + bias[oc];
    // Hardswish: x * relu6(x + 3) / 6
    float hs = val * fminf(fmaxf(val + 3.0f, 0.0f), 6.0f) / 6.0f;
    // ReLU
    output[((b * out_channels + oc) * out_h + oh) * out_w + ow] = fmaxf(hs, 0.0f);
}

void fused_conv_activation_forward(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
    torch::Tensor& output, int stride, int padding, int dilation, int groups
) {
    int B = input.size(0);
    int OC = weight.size(0);
    int OH = output.size(2);
    int OW = output.size(3);

    dim3 block(1, 1, 1); // Simple grid mapping
    dim3 grid(OW, OH, B * OC);
    
    fused_conv_activation_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), B, input.size(1), OC, 
        input.size(2), input.size(3), weight.size(2),
        stride, padding, dilation, groups
    );
}
"""

cpp_source = r"""
void fused_conv_activation_forward(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
    torch::Tensor& output, int stride, int padding, int dilation, int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward", &fused_conv_activation_forward, "Fused conv+hs+relu");
}
"""

fused_ext = load_inline(
    name='fused_module',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    out_h = (x.shape[2] + 2 * conv_padding - conv_dilation * (conv_weight.shape[2] - 1) - 1) // conv_stride + 1
    out_w = (x.shape[3] + 2 * conv_padding - conv_dilation * (conv_weight.shape[3] - 1) - 1) // conv_stride + 1
    output = torch.empty((x.size(0), conv_weight.size(0), out_h, out_w), device=x.device)
    
    fused_ext.fused_forward(
        x, conv_weight, conv_bias, output,
        conv_stride, conv_padding, conv_dilation, conv_groups
    )
    return output

batch_size, in_channels, out_channels, height, width, kernel_size = 128, 8, 64, 128, 128, 3
def get_init_inputs(): return [in_channels, out_channels, kernel_size]
def get_inputs(): return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
