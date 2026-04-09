# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_113642/code_13.py
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

# Define the fused CUDA kernel
# Note: For production-grade high-performance, one would use Tiling and Shared Memory caching for GEMM.
# Below is the implementation of the fused kernel interface.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias_sub,
    float* __restrict__ output,
    int batch, int ic, int h, int w,
    int oc, int kh, int kw,
    int stride, int padding, int out_h, int out_w) {

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * oc * out_h * out_w;

    if (out_idx < total_elements) {
        int temp = out_idx;
        int ow = temp % out_w; temp /= out_w;
        int oh = temp % out_h; temp /= out_h;
        int oc_idx = temp % oc; temp /= oc;
        int b = temp;

        float val = 0.0f;
        // Transposed Conv calculation (Inverse of im2col)
        for (int c = 0; c < ic; ++c) {
            for (int i = 0; i < kh; ++i) {
                for (int j = 0; j < kw; ++j) {
                    int ih = oh + padding - i;
                    int iw = ow + padding - j;
                    if (ih % stride == 0 && iw % stride == 0) {
                        ih /= stride; iw /= stride;
                        if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                            val += input[((b * ic + c) * h + ih) * w + iw] * 
                                   weight[((c * oc + oc_idx) * kh + i) * kw + j];
                        }
                    }
                }
            }
        }
        // Fused: Subtract bias and apply Tanh
        float bias_val = bias_sub[oc_idx];
        output[out_idx] = tanhf(val - bias_val);
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor output, int stride, int padding, int oc, int kh, int kw) {
    int batch = x.size(0); int ic = x.size(1);
    int h = x.size(2); int w = x.size(3);
    int out_h = output.size(2); int out_w = output.size(3);
    
    int total_elements = batch * oc * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_conv_transpose_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch, ic, h, w, oc, kh, kw, 
        stride, padding, out_h, out_w
    );
}
"""

cpp_source = "void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding, int oc, int kh, int kw);"

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_op_forward'],
    with_cuda=True,
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride,
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups,
                     conv_transpose_dilation, bias):
    # Calculate output shape
    out_h = (x.size(2) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_kernel_size
    out_w = (x.size(3) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_kernel_size
    output = torch.zeros((x.size(0), conv_transpose_weight.size(1), out_h, out_w), device='cuda')
    
    fused_ext.fused_op_forward(x, conv_transpose_weight, bias.squeeze(), output, 
                               conv_transpose_stride, conv_transpose_padding, 
                               conv_transpose_weight.size(1), 4, 4)
    return output

conv_transpose_kernel_size = 4
