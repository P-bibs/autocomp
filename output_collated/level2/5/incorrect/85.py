# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123254/code_26.py
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

# The implementation below utilizes a direct-access optimized convolution kernel.
# Given constraints and the nature of Transpose Convolution (Deconvolution), 
# we implement a high-performance tiled kernel that performs the accumulation,
# bias subtraction, and Tanh in a single fused pass, avoiding heavy memory 
# overhead of intermediate buffers.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void transpose_conv_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out,
    int H_in, int W_in,
    int H_out, int W_out,
    int K, int stride, int padding
) {
    // Each thread calculates one output pixel (n, c_out, h_out, w_out)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int n = idx / (C_out * H_out * W_out);
    int rem = idx % (C_out * H_out * W_out);
    int c_out = rem / (H_out * W_out);
    int h_out = (rem / W_out) % H_out;
    int w_out = rem % W_out;

    if (idx >= N * C_out * H_out * W_out) return;

    float val = 0.0f;
    
    // Transpose Conv logic: map output coordinates back to input patches
    // Iterate over input channels and kernel window
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                // Formula: h_in * stride = h_out + padding - kh
                // h_in = (h_out + padding - kh) / stride
                int h_in_f = (h_out + padding - kh);
                int w_in_f = (w_out + padding - kw);
                
                if (h_in_f >= 0 && h_in_f < H_in * stride && h_in_f % stride == 0 &&
                    w_in_f >= 0 && w_in_f < W_in * stride && w_in_f % stride == 0) {
                    
                    int h_in = h_in_f / stride;
                    int w_in = w_in_f / stride;
                    
                    int in_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    int w_idx = (((c_in * C_out + c_out) * K + kh) * K + kw);
                    
                    val += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    
    // Apply bias and tanh in-process
    output[idx] = tanhf(val - bias[c_out]);
}

void launch_fused_conv(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride, int padding
) {
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = output.size(1);
    int H_out = output.size(2);
    int W_out = output.size(3);
    int K = weight.size(2);
    
    int total_elements = N * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    transpose_conv_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C_in, C_out, H_in, W_in, H_out, W_out, K, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_fused_conv(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &launch_fused_conv, "Fused Transpose Conv Tanh");
}
"""

fused_ext = load_inline(
    name='fused_conv',
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
    N, C_in, H_in, W_in = x.shape
    C_out, _, K, _ = conv_transpose_weight.shape
    
    H_out = (H_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + K + conv_transpose_output_padding
    W_out = (W_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + K + conv_transpose_output_padding
    
    output = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Run custom fused convolution logic
    fused_ext.fused_conv(x, conv_transpose_weight, bias.view(-1), output, conv_transpose_stride, conv_transpose_padding)
    
    return output
