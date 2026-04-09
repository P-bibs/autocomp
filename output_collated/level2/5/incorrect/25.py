# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_114641/code_8.py
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

# The custom manual convolution transpose kernel
# Using a 2D grid/block strategy for better occupancy on RTX 2080Ti
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_transpose_activation_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ act_bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_size, int stride, int padding
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;

    if (out_idx >= total_elements) return;

    int tmp = out_idx;
    int ow = tmp % out_width; tmp /= out_width;
    int oh = tmp % out_height; tmp /= out_height;
    int oc = tmp % out_channels; tmp /= out_channels;
    int batch = tmp;

    float sum = (conv_bias != nullptr) ? conv_bias[oc] : 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            int ih_full = oh + padding - kh;
            if (ih_full % stride != 0) continue;
            int ih = ih_full / stride;
            if (ih < 0 || ih >= in_height) continue;

            for (int kw = 0; kw < kernel_size; ++kw) {
                int iw_full = ow + padding - kw;
                if (iw_full % stride != 0) continue;
                int iw = iw_full / stride;
                if (iw < 0 || iw >= in_width) continue;

                int input_idx = ((batch * in_channels + ic) * in_height + ih) * in_width + iw;
                int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    output[out_idx] = tanhf(sum - act_bias[oc]);
}

void launch_fused_kernel(
    const torch::Tensor& input, const torch::Tensor& weight, 
    const torch::Tensor& conv_bias, const torch::Tensor& act_bias, 
    torch::Tensor& output, int stride, int padding
) {
    int N = input.size(0);
    int IC = input.size(1);
    int iH = input.size(2);
    int iW = input.size(3);
    int OC = weight.size(0);
    int KH = weight.size(2);
    int oH = output.size(2);
    int oW = output.size(3);

    int total = N * OC * oH * oW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    const float* c_bias_ptr = conv_bias.defined() ? conv_bias.data_ptr<float>() : nullptr;

    fused_conv_transpose_activation_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), c_bias_ptr,
        act_bias.data_ptr<float>(), output.data_ptr<float>(),
        N, IC, OC, iH, iW, oH, oW, KH, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_fused_kernel(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& conv_bias, const torch::Tensor& act_bias, torch::Tensor& output, int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused_kernel, "Fused ConvTranspose + Bias + Tanh");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride,
    conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups,
    conv_transpose_dilation, bias
):
    N, IC, iH, iW = x.shape
    OC, _, KH, KW = conv_transpose_weight.shape
    oH = (iH - 1) * conv_transpose_stride - 2 * conv_transpose_padding + KH + conv_transpose_output_padding
    oW = (iW - 1) * conv_transpose_stride - 2 * conv_transpose_padding + KW + conv_transpose_output_padding
    
    output = torch.empty((N, OC, oH, oW), device=x.device, dtype=x.dtype)
    
    # Empty tensor for optional conv_bias
    c_bias = conv_transpose_bias if conv_transpose_bias is not None else torch.tensor([], device=x.device)
    
    fused_ext.fused_op(
        x, conv_transpose_weight, c_bias, bias.view(-1), output,
        conv_transpose_stride, conv_transpose_padding
    )
    return output
