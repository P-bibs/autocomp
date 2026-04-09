# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112403/code_5.py
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

# CUDA kernel: Parallel implementation of ConvTranspose2d + Bias Subtract + Tanh
cuda_source = r"""
#include <torch/extension.h>

__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out, int H_in, int W_in,
    int H_out, int W_out, int kernel_size, int stride, int padding) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * H_out * W_out;
    if (tid >= total_elements) return;

    // Map linear index to 4D coordinates
    int ow = tid % W_out;
    int oh = (tid / W_out) % H_out;
    int oc = (tid / (W_out * H_out)) % C_out;
    int n = tid / (W_out * H_out * C_out);

    float val = 0.0f;
    // ConvTranspose logic: iterate over input and filter
    for (int ic = 0; ic < C_in; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = (oh + padding - kh) / stride;
                int iw = (ow + padding - kw) / stride;
                
                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in &&
                    (oh + padding - kh) % stride == 0 && (ow + padding - kw) % stride == 0) {
                    val += x[((n * C_in + ic) * H_in + ih) * W_in + iw] * 
                           weight[((ic * C_out + oc) * kernel_size + kh) * kernel_size + kw];
                }
            }
        }
    }
    output[tid] = tanhf(val - bias[oc]);
}

void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, 
                      int stride, int padding) {
    int N = x.size(0); int C_in = x.size(1);
    int H_in = x.size(2); int W_in = x.size(3);
    int C_out = weight.size(1);
    int H_out = output.size(2); int W_out = output.size(3);
    int kernel_size = weight.size(2);

    int total_elements = N * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_transpose_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, C_in, C_out, H_in, W_in, H_out, W_out, kernel_size, stride, padding);
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused ConvTranspose2d Bias Tanh");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride,
                     conv_transpose_padding, conv_transpose_output_padding,
                     conv_transpose_groups, conv_transpose_dilation, bias):
    # Output dimensions for kernel 4x4 stride 1 padding 0
    h_out = (x.size(2) - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + 4
    w_out = (x.size(3) - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + 4
    
    # Weight shape expectation: [in_channels, out_channels, k, k]
    weight = conv_transpose_weight 
    b = bias.view(-1)
    output = torch.empty((x.size(0), weight.size(1), h_out, w_out), device=x.device)
    
    fused_ext.fused_op(x, weight, b, output, conv_transpose_stride[0], conv_transpose_padding[0])
    return output
