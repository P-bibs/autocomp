# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_113642/code_14.py
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

# ----------------------------------------------------------------------
# CUDA source: custom transposed-convolution kernel fused with bias + tanh
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void deconv_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in,
    const int H_out, const int W_out,
    const int K_h, const int K_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const int groups)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    // Decode flat index
    const int c_out = (idx / (W_out * H_out)) % C_out;
    const int w_out = idx % W_out;
    const int h_out = (idx / W_out) % H_out;
    const int n     = idx / (C_out * H_out * W_out);

    float acc = 0.0f;

    const int group_size_in = C_in / groups;
    const int group_size_out = C_out / groups;
    const int group_id = c_out / group_size_out;
    const int c_in_start = group_id * group_size_in;
    const int c_in_end = c_in_start + group_size_in;

    // Direct implementation of Transposed Convolution: 
    // Output pixel (h_out, w_out) is influenced by sliding window.
    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        for (int kh = 0; kh < K_h; ++kh) {
            int y_in_raw = h_out + pad_h - kh * dilation_h;
            if (y_in_raw % stride_h != 0) continue;
            int y_in = y_in_raw / stride_h;
            if (y_in < 0 || y_in >= H_in) continue;

            for (int kw = 0; kw < K_w; ++kw) {
                int x_in_raw = w_out + pad_w - kw * dilation_w;
                if (x_in_raw % stride_w != 0) continue;
                int x_in = x_in_raw / stride_w;
                if (x_in < 0 || x_in >= W_in) continue;

                // Weight layout: [C_in, C_out, K_h, K_w]
                int w_idx = ((c_in * C_out + c_out) * K_h + kh) * K_w + kw;
                int in_idx = ((n * C_in + c_in) * H_in + y_in) * W_in + x_in;
                
                acc += __ldg(&input[in_idx]) * __ldg(&weight[w_idx]);
            }
        }
    }

    acc += __ldg(&bias[c_out]);
    output[idx] = tanhf(acc);
}

void deconv_fused(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int N, int C_in, int C_out, int H_in, int W_in, int H_out, int W_out,
    int K_h, int K_w, int stride_h, int stride_w, int pad_h, int pad_w,
    int dilation_h, int dilation_w, int groups)
{
    const int total = N * C_out * H_out * W_out;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    deconv_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C_in, C_out, H_in, W_in, H_out, W_out,
        K_h, K_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups);
}
"""

cpp_source = r"""
void deconv_fused(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                  int N, int C_in, int C_out, int H_in, int W_in, int H_out, int W_out,
                  int K_h, int K_w, int stride_h, int stride_w, int pad_h, int pad_w,
                  int dilation_h, int dilation_w, int groups);
"""

fused_ext = load_inline(
    name='fused_deconv',
    cpp_sources=cpp_source + "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(" + '"deconv_fused"' + ", &deconv_fused); }",
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride,
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups,
                     conv_transpose_dilation, bias):
    x = x.cuda().contiguous()
    w = conv_transpose_weight.cuda().contiguous()
    b = bias.view(-1).cuda().contiguous()
    
    N, C_in, H_in, W_in = x.shape
    C_out, K_h, K_w = w.size(1), w.size(2), w.size(3)
    
    s = conv_transpose_stride if isinstance(conv_transpose_stride, int) else conv_transpose_stride[0]
    p = conv_transpose_padding if isinstance(conv_transpose_padding, int) else conv_transpose_padding[0]
    op = conv_transpose_output_padding if isinstance(conv_transpose_output_padding, int) else conv_transpose_output_padding[0]
    d = conv_transpose_dilation if isinstance(conv_transpose_dilation, int) else conv_transpose_dilation[0]
    
    H_out = (H_in - 1) * s - 2 * p + d * (K_h - 1) + op + 1
    W_out = (W_in - 1) * s - 2 * p + d * (K_w - 1) + op + 1
    
    out = torch.empty((N, C_out, H_out, W_out), device='cuda', dtype=x.dtype)
    fused_ext.deconv_fused(x, w, b, out, N, C_in, C_out, H_in, W_in, H_out, W_out,
                           K_h, K_w, s, s, p, p, d, d, conv_transpose_groups)
    return out
