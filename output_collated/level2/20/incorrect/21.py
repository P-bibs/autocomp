# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_130013/code_20.py
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

# CUDA kernel with custom Transpose Conv 3D (Direct approach) + Fused Activation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int C_out, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out, int K, int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = B * C_out * D_out * H_out * W_out;
    if (idx >= total_out) return;

    int tmp = idx;
    int w_out = tmp % W_out; tmp /= W_out;
    int h_out = tmp % H_out; tmp /= H_out;
    int d_out = tmp % D_out; tmp /= D_out;
    int c_out = tmp % C_out; tmp /= C_out;
    int b = tmp;

    float acc = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kd = 0; kd < K; ++kd) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int d_in = (d_out + padding - kd) / stride;
                    int h_in = (h_out + padding - kh) / stride;
                    int w_in = (w_out + padding - kw) / stride;

                    if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in &&
                        (d_out + padding - kd) % stride == 0 &&
                        (h_out + padding - kh) % stride == 0 &&
                        (w_out + padding - kw) % stride == 0) {
                        
                        acc += input[((b * C_in + c_in) * D_in + d_in) * H_in * W_in + h_in * W_in + w_in] *
                               weight[((c_in * C_out + c_out) * K + kd) * K * K + kh * K + kw];
                    }
                }
            }
        }
    }
    
    float y = acc; // Result of conv_transpose3d
    float b_val = bias[c_out];
    // Mathematical simplification: output = 2*y^2 + (b+1)*y
    output[idx] = 2.0f * y * y + (b_val + 1.0f) * y;
}

void launch_fused_conv(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, 
                       int stride, int padding, int K) {
    int B = input.size(0), C_in = input.size(1), D_in = input.size(2), H_in = input.size(3), W_in = input.size(4);
    int C_out = weight.size(1);
    int D_out = output.size(2), H_out = output.size(3), W_out = output.size(4);
    
    int total = B * C_out * D_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    fused_conv_transpose_bias_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        B, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out, K, stride, padding
    );
}
"""

cpp_source = r"""
void launch_fused_conv(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding, int K);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused_conv", &launch_fused_conv); }
"""

fused_ext = load_inline(name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_kernel, with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, conv_transpose_padding, 
                     conv_transpose_output_padding, conv_transpose_groups, conv_transpose_dilation, bias):
    # Output shape calculation for Transpose Conv
    batch = x.shape[0]
    out_c = conv_transpose_weight.shape[1]
    d_out = (x.shape[2] - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.shape[2] + conv_transpose_output_padding
    h_out, w_out = [(s - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.shape[3] + conv_transpose_output_padding for s in [x.shape[3], x.shape[4]]]
    
    output = torch.empty((batch, out_c, d_out, h_out, w_out), device=x.device)
    fused_ext.fused_conv(x.contiguous(), conv_transpose_weight.contiguous(), bias.flatten(), output, 
                         conv_transpose_stride, conv_transpose_padding, conv_transpose_weight.shape[2])
    return output
