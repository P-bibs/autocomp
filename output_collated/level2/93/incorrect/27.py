# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_153448/code_15.py
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

# ----------------------------------------------------------------------
# CUDA source – Fused Transposed Conv2D + Pointwise ops
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void deconv_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int H, const int W,
    const int C_out, const int K,
    const int stride, const int padding, const int dilation,
    const int out_h, const int out_w,
    const float add_value,
    const float multiply_value,
    const bool has_bias)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * out_h * out_w;
    if (idx >= total) return;

    // Decode linear index to (n, oc, oh, ow)
    int tmp = idx;
    int ow = tmp % out_w; tmp /= out_w;
    int oh = tmp % out_h; tmp /= out_h;
    int oc = tmp % C_out; tmp /= C_out;
    int n  = tmp;

    float sum = has_bias ? bias[oc] : 0.0f;

    // Naive Transposed Convolution
    // Each output pixel corresponds to a kernel sliding over the input
    for (int ic = 0; ic < C_in; ++ic) {
        for (int ky = 0; ky < K; ++ky) {
            int i_h = (oh + padding - ky * dilation);
            if (i_h % stride != 0) continue;
            i_h /= stride;
            if (i_h < 0 || i_h >= H) continue;

            for (int kx = 0; kx < K; ++kx) {
                int i_w = (ow + padding - kx * dilation);
                if (i_w % stride != 0) continue;
                i_w /= stride;
                if (i_w < 0 || i_w >= W) continue;

                float inp_val = input[((n * C_in + ic) * H + i_h) * W + i_w];
                float w_val = weight[((ic * C_out + oc) * K + ky) * K + kx];
                sum += inp_val * w_val;
            }
        }
    }

    // Fused point-wise operations
    float val = sum + add_value;
    val = (val < 0.0f) ? val : 0.0f; // min(x, 0)
    
    // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float x = val;
    float tanh_arg = 0.7978845608f * (x + 0.044715f * x * x * x);
    val = 0.5f * x * (1.0f + tanhf(tanh_arg));
    
    output[((n * C_out + oc) * out_h + oh) * out_w + ow] = val * multiply_value;
}

void deconv_fused_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride, int padding, int dilation,
    float add_value, float multiply_value)
{
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int C_out = weight.size(1);
    const int K = weight.size(2);
    const int out_h = output.size(2);
    const int out_w = output.size(3);

    const int total = N * C_out * out_h * out_w;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    deconv_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr, output.data_ptr<float>(),
        N, C_in, H, W, C_out, K, stride, padding, dilation, out_h, out_w,
        add_value, multiply_value, bias.defined());
}
"""

cpp_source = r"""
void deconv_fused_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride, int padding, int dilation,
    float add_value, float multiply_value);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &deconv_fused_forward, "Fused deconv operation");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride,
    conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups,
    conv_transpose_dilation, add_value, multiply_value,
):
    # Ensure inputs are GPU tensors
    x = x.to(device='cuda', dtype=torch.float32)
    w = conv_transpose_weight.to(device='cuda', dtype=torch.float32)
    b = conv_transpose_bias.to(device='cuda', dtype=torch.float32) if conv_transpose_bias is not None else None
    
    N, C_in, H, W = x.shape
    C_out, K = w.size(1), w.size(2)
    out_h = (H - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    out_w = (W - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    
    output = torch.empty((N, C_out, out_h, out_w), device='cuda', dtype=torch.float32)
    fused_ext.fused_op(x, w, b, output, conv_transpose_stride, conv_transpose_padding, conv_transpose_dilation, add_value, multiply_value)
    return output
