# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_121229/code_6.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fused_conv_transpose_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int in_c, int out_c, 
    int in_h, int in_w, int out_h, int out_w, 
    int kh, int kw) 
{
    int n = blockIdx.z;
    int oc = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_idx >= out_h * out_w) return;
    
    int oh = out_idx / out_w;
    int ow = out_idx % out_w;
    
    float acc = 0.0f;
    
    // Perform transposed convolution (gradient-like approach)
    for (int ic = 0; ic < in_c; ++ic) {
        for (int i = 0; i < kh; ++i) {
            for (int j = 0; j < kw; ++j) {
                int ih = oh - i;
                int iw = ow - j;
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    float w_val = weight[((ic * out_c + oc) * kh + i) * kw + j];
                    float x_val = input[((n * in_c + ic) * in_h + ih) * in_w + iw];
                    acc += x_val * w_val;
                }
            }
        }
    }
    
    // Apply bias and Tanh fused
    output[((n * out_c + oc) * out_h + oh) * out_w + ow] = tanhf(acc + bias[oc]);
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output
) {
    const int N = input.size(0);
    const int in_c = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_c = weight.size(1);
    const int kh = weight.size(2);
    const int kw = weight.size(3);
    const int out_h = output.size(2);
    const int out_w = output.size(3);

    dim3 threads(256);
    dim3 blocks((out_h * out_w + threads.x - 1) / threads.x, out_c, N);

    fused_conv_transpose_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, in_c, out_c, in_h, in_w, out_h, out_w, kh, kw
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward, "Fused ConvTranspose2d + Bias + Tanh");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    bias,
):
    # Validate that we're using supported parameters
    assert conv_transpose_stride == 1, "Only stride=1 is supported"
    assert conv_transpose_padding == 0, "Only padding=0 is supported"
    assert conv_transpose_output_padding == 0, "Only output_padding=0 is supported"
    assert conv_transpose_groups == 1, "Only groups=1 is supported"
    assert conv_transpose_dilation == 1, "Only dilation=1 is supported"
    
    N, in_c, in_h, in_w = x.shape
    out_c, _, kh, kw = conv_transpose_weight.shape
    
    # Calculate output dimensions for transposed convolution with stride=1, padding=0
    out_h = in_h + kh - 1
    out_w = in_w + kw - 1
    
    # Create output tensor
    output = torch.empty((N, out_c, out_h, out_w), device=x.device, dtype=x.dtype)
    
    # Run fused kernel
    fused_ext.fused_op_forward(x, conv_transpose_weight, conv_transpose_bias, output)
    
    # Apply final bias subtraction and tanh
    bias_flat = bias.view(1, -1, 1, 1)
    output = torch.tanh(output - bias_flat)
    
    return output
