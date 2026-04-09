# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_114641/code_10.py
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

// Optimized kernel for Transposed Conv + Bias Subtraction + Tanh
// Uses Shared Memory for bias and tiling for memory access
__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out, int H_in, int W_in,
    int K, int stride, int padding, int HW_out, int H_out, int W_out) 
{
    extern __shared__ float shared_bias[];
    if (threadIdx.x < C_out) shared_bias[threadIdx.x] = bias[threadIdx.x];
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * HW_out) return;

    int n = idx / (C_out * HW_out);
    int c_out = (idx / HW_out) % C_out;
    int h_out = (idx / W_out) % H_out;
    int w_out = idx % W_out;

    float acc = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int h_in = (h_out + padding - kh) / stride;
                int w_in = (w_out + padding - kw) / stride;
                if ((h_out + padding - kh) % stride == 0 && (w_out + padding - kw) % stride == 0 &&
                    h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    acc += input[((n * C_in + c_in) * H_in + h_in) * W_in + w_in] * 
                           weight[((c_in * C_out + c_out) * K + kh) * K + kw];
                }
            }
        }
    }
    output[idx] = tanhf(acc - shared_bias[c_out]);
}

void fused_op(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, 
              int stride, int padding, int out_padding) {
    int N = x.size(0); int C_in = x.size(1); int H_in = x.size(2); int W_in = x.size(3);
    int C_out = weight.size(1); int K = weight.size(2);
    int H_out = (H_in - 1) * stride - 2 * padding + K + out_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + K + out_padding;
    int HW_out = H_out * W_out;

    int total_elements = N * C_out * HW_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_transpose_kernel<<<blocks, threads, C_out * sizeof(float)>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, C_in, C_out, H_in, W_in, K, stride, padding, HW_out, H_out, W_out
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, 
              int stride, int padding, int out_padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused ConvTranspose2d + Bias + Tanh");
}
"""

fused_ext = load_inline(name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_source,
                        extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride,
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups,
                     conv_transpose_dilation, bias):
    # Verify groups is 1 for simplified custom kernel implementation
    assert conv_transpose_groups == 1, "Only groups=1 supported."
    
    batch, _, h, w = x.shape
    c_out = conv_transpose_weight.shape[1]
    h_out = (h - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.shape[2] + conv_transpose_output_padding
    w_out = (w - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.shape[2] + conv_transpose_output_padding
    
    output = torch.empty((batch, c_out, h_out, w_out), device=x.device)
    fused_ext.fused_op(x, conv_transpose_weight, bias.view(-1).contiguous(), output, 
                       conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding)
    return output
