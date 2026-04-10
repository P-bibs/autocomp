# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152044/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'add_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'add_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

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
    if 'add_value' in flat_state:
        state_kwargs['add_value'] = flat_state['add_value']
    else:
        state_kwargs['add_value'] = getattr(model, 'add_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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

# -------------------------------------------------------------------------
# CUDA implementation of a fused Transposed Convolution + Activation kernel
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

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
    const int groups,
    const float add_value,
    const float multiply_value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    // Index mapping (n, c, y, x)
    int remaining = idx;
    int x_out = remaining % W_out; remaining /= W_out;
    int y_out = remaining % H_out; remaining /= H_out;
    int c_out = remaining % C_out; remaining /= C_out;
    int n = remaining;

    const int C_in_per_group = C_in / groups;
    const int group_id = c_out / (C_out / groups);
    const int c_in_start = group_id * C_in_per_group;

    float acc = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Transposed Conv math: sum over filter and inputs
    for (int cc = 0; cc < C_in_per_group; ++cc) {
        int c_in = c_in_start + cc;
        for (int ky = 0; ky < K_h; ++ky) {
            int y_in = (y_out + pad_h - ky);
            if (y_in < 0 || y_in % stride_h != 0) continue;
            y_in /= stride_h;
            if (y_in >= H_in) continue;

            for (int kx = 0; kx < K_w; ++kx) {
                int x_in = (x_out + pad_w - kx);
                if (x_in < 0 || x_in % stride_w != 0) continue;
                x_in /= stride_w;
                if (x_in >= W_in) continue;

                float in_val = input[((n * C_in + c_in) * H_in + y_in) * W_in + x_in];
                float w_val = weight[((c_in * (C_out / groups) + (c_out % (C_out / groups))) * K_h + ky) * K_w + kx];
                acc += in_val * w_val;
            }
        }
    }

    // Fused element-wise operations:
    // 1. Add
    acc += add_value;
    // 2. ReLU
    if (acc < 0.0f) acc = 0.0f;
    // 3. GELU (TanH approx)
    acc = 0.5f * acc * (1.0f + tanhf(0.7978845608f * (acc + 0.044715f * acc * acc * acc)));
    // 4. Multiply
    output[idx] = acc * multiply_value;
}

void fused_op_forward(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
    torch::Tensor& output, int N, int C_in, int C_out, int H_in, int W_in,
    int H_out, int W_out, int K_h, int K_w, int stride_h, int stride_w,
    int pad_h, int pad_w, int groups, float add_value, float multiply_value)
{
    const int total = N * C_out * H_out * W_out;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    deconv_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        (bias.numel() > 0 ? bias.data_ptr<float>() : nullptr),
        output.data_ptr<float>(), N, C_in, C_out, H_in, W_in, H_out, W_out,
        K_h, K_w, stride_h, stride_w, pad_h, pad_w, groups, add_value, multiply_value
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& output, int N, int C_in, int C_out, int H_in, int W_in, int H_out, int W_out, int K_h, int K_w, int stride_h, int stride_w, int pad_h, int pad_w, int groups, float add_value, float multiply_value);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused_op", &fused_op_forward, "Fused Deconv+Ops"); }
"""

fused_ext = load_inline(name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source, extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, conv_transpose_dilation, add_value, multiply_value):
    N, C_in, H_in, W_in = x.shape
    C_in_w, C_out_div_groups, K_h, K_w = conv_transpose_weight.shape
    C_out = C_out_div_groups * conv_transpose_groups
    
    stride = conv_transpose_stride if isinstance(conv_transpose_stride, int) else conv_transpose_stride[0]
    pad = conv_transpose_padding if isinstance(conv_transpose_padding, int) else conv_transpose_padding[0]
    out_pad = conv_transpose_output_padding if isinstance(conv_transpose_output_padding, int) else conv_transpose_output_padding[0]
    
    H_out = (H_in - 1) * stride - 2 * pad + (K_h - 1) + out_pad + 1
    W_out = (W_in - 1) * stride - 2 * pad + (K_w - 1) + out_pad + 1
    
    output = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    bias = conv_transpose_bias if conv_transpose_bias is not None else torch.empty(0, device=x.device)
    
    fused_ext.fused_op(x, conv_transpose_weight, bias, output, N, C_in, C_out, H_in, W_in, H_out, W_out, K_h, K_w, stride, stride, pad, pad, conv_transpose_groups, float(add_value), float(multiply_value))
    return output
