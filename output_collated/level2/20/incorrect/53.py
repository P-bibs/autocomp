# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_133559/code_18.py
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

# Combined kernel handling both Convolution Transpose and the Fused Post-Processing
# This minimizes global memory round-trips by keeping data in registers/L1 cache.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void fused_conv_post_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int kernel_size, int stride, int padding
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch * out_channels * out_depth * out_height * out_width;
    
    if (tid >= total_output_elements) return;

    // Map linear index to output coordinates
    int temp = tid;
    int w = temp % out_width; temp /= out_width;
    int h = temp % out_height; temp /= out_height;
    int d = temp % out_depth; temp /= out_depth;
    int oc = temp % out_channels;
    int b = temp / out_channels;

    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            int in_d = (d + padding - kd);
            if (in_d % stride != 0) continue;
            int i_d = in_d / stride;
            if (i_d < 0 || i_d >= in_depth) continue;

            for (int kh = 0; kh < kernel_size; ++kh) {
                int in_h = (h + padding - kh);
                if (in_h % stride != 0) continue;
                int i_h = in_h / stride;
                if (i_h < 0 || i_h >= in_height) continue;

                for (int kw = 0; kw < kernel_size; ++kw) {
                    int in_w = (w + padding - kw);
                    if (in_w % stride != 0) continue;
                    int i_w = in_w / stride;
                    if (i_w < 0 || i_w >= in_width) continue;

                    int input_idx = (((b * in_channels + ic) * in_depth + i_d) * in_height + i_h) * in_width + i_w;
                    int weight_idx = (((oc * in_channels + ic) * kernel_size + kd) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Fused operation: ((sum + bias) + sum) * sum + sum
    float b_val = bias[oc];
    float val = ((sum + b_val) + sum) * sum + sum;
    output[tid] = val;
}

void fused_forward(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
    torch::Tensor& output, int stride, int padding
) {
    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_channels = weight.size(1);
    int kernel_size = weight.size(3);
    
    int out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size;
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size;

    int total_elements = batch * out_channels * out_depth * out_height * out_width;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_conv_post_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch, in_channels, out_channels,
        in_depth, in_height, in_width, out_depth, out_height, out_width,
        kernel_size, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_forward(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& output, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward", &fused_forward, "Fused Transpose Conv and Post-op");
}
"""

module = load_inline(
    name='fused_conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x, *, conv_transpose_weight, conv_transpose_bias,
    conv_transpose_stride, conv_transpose_padding,
    conv_transpose_output_padding, conv_transpose_groups,
    conv_transpose_dilation, bias
):
    batch, in_channels, in_depth, in_height, in_width = x.shape
    out_channels = conv_transpose_weight.size(1)
    kernel_size = conv_transpose_weight.size(3)
    # Adjust dims based on stride/padding logic
    stride = conv_transpose_stride[0]
    padding = conv_transpose_padding[0]
    out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + conv_transpose_output_padding[0]
    out_height = (in_height - 1) * stride - 2 * padding + kernel_size + conv_transpose_output_padding[0]
    out_width = (in_width - 1) * stride - 2 * padding + kernel_size + conv_transpose_output_padding[0]
    
    output = torch.empty((batch, out_channels, out_depth, out_height, out_width), device=x.device)
    module.fused_forward(x.contiguous(), conv_transpose_weight.contiguous(), bias.view(-1), output, stride, padding)
    return output
