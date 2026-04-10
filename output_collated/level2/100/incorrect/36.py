# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_132110/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'min_value', 'divisor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'min_value', 'divisor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

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
    if 'min_value' in flat_state:
        state_kwargs['min_value'] = flat_state['min_value']
    else:
        state_kwargs['min_value'] = getattr(model, 'min_value')
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
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

# ------------------------------------------------------------
# 1.  CUDA kernel (fused conv_transpose3d + clamp + scale)
# ------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,          // may be nullptr
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int K, const int stride, const int padding,
    const int output_padding,
    const int groups, const int dilation,
    const float min_value,
    const float inv_divisor)
{
    const int total_out = N * C_out * D_out * H_out * W_out;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    // ----- decode flat index to (n, oc, od, oh, ow) -----
    int tmp = idx;
    int n = tmp / (C_out * D_out * H_out * W_out);
    tmp %= (C_out * D_out * H_out * W_out);
    int oc = tmp / (D_out * H_out * W_out);
    tmp %= (D_out * H_out * W_out);
    int od = tmp / (H_out * W_out);
    tmp %= (H_out * W_out);
    int oh = tmp / W_out;
    int ow = tmp % W_out;

    // ----- accumulate contribution -----
    float sum = 0.0f;
    if (bias != nullptr) sum += bias[oc];

    // loop over input channels
    for (int ic = 0; ic < C_in; ++ic) {
        // base offset for weight[oc][ic][*]
        const int wbase = ((oc * C_in + ic) * K * K * K);

        for (int kd = 0; kd < K; ++kd) {
            int d_in = (od + padding - kd - output_padding);
            if (d_in < 0) continue;
            if (d_in % stride != 0) continue;
            d_in /= stride;
            if (d_in >= D_in) continue;

            for (int kh = 0; kh < K; ++kh) {
                int h_in = (oh + padding - kh - output_padding);
                if (h_in < 0) continue;
                if (h_in % stride != 0) continue;
                h_in /= stride;
                if (h_in >= H_in) continue;

                for (int kw = 0; kw < K; ++kw) {
                    int w_in = (ow + padding - kw - output_padding);
                    if (w_in < 0) continue;
                    if (w_in % stride != 0) continue;
                    w_in /= stride;
                    if (w_in >= W_in) continue;

                    // flat index for input tensor
                    int in_idx = (((n * C_in + ic) * D_in + d_in) * H_in + h_in) * W_in + w_in;
                    float v_in = input[in_idx];

                    int w_idx = wbase + (kd * K * K + kh * K + kw);
                    float v_w = weight[w_idx];

                    sum += v_in * v_w;
                }
            }
        }
    }

    // ----- clamp & scale -----
    if (sum < min_value) sum = min_value;
    sum *= inv_divisor;

    // ----- write result -----
    int out_idx = (((n * C_out + oc) * D_out + od) * H_out + oh) * W_out + ow;
    output[out_idx] = sum;
}

void fused_conv_transpose3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int K, int stride, int padding,
    int output_padding,
    int groups, int dilation,
    float min_value,
    float inv_divisor)
{
    const float* in_ptr   = input.data_ptr<float>();
    const float* w_ptr    = weight.data_ptr<float>();
    const float* b_ptr    = bias.numel() ? bias.data_ptr<float>() : nullptr;
    float* out_ptr        = output.data_ptr<float>();

    const int threads = 256;
    const int total_out = N * C_out * D_out * H_out * W_out;
    const int blocks = (total_out + threads - 1) / threads;

    fused_conv_transpose3d_kernel<<<blocks, threads>>>(
        in_ptr, w_ptr, b_ptr, out_ptr,
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        K, stride, padding, output_padding,
        groups, dilation,
        min_value, inv_divisor);
}
"""

# ------------------------------------------------------------
# 2.  C++ binding (PYBIND11)
# ------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int K, int stride, int padding,
    int output_padding,
    int groups, int dilation,
    float min_value,
    float inv_divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_forward",
          &fused_conv_transpose3d_forward,
          "Fused transposed 3D convolution + clamp + scale");
}
"""

# ------------------------------------------------------------
# 3.  Compile the extension (runs once)
# ------------------------------------------------------------
fused_ext = load_inline(
    name='fused_conv_transpose3d',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ------------------------------------------------------------
# 4.  The functional model used by the benchmark
# ------------------------------------------------------------
def functional_model(
    x,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    min_value,
    divisor,
):
    # Move everything to the current device (assumed to be CUDA)
    x = x.cuda()
    weight = conv_transpose_weight.cuda()
    if conv_transpose_bias is not None:
        bias = conv_transpose_bias.cuda()
    else:
        # empty tensor – the kernel treats a nullptr as “no bias”
        bias = torch.empty(0, dtype=x.dtype, device=x.device)

    N, C_in, D_in, H_in, W_in = x.shape
    out_channels = weight.shape[0]
    K = weight.shape[2]          # kernel size (assumed square)

    # Compute output spatial size using the transposed-conv formula
    D_out = (D_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + \
            conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    H_out = (H_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + \
            conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    W_out = (W_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + \
            conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1

    # Allocate output tensor
    output = torch.empty((N, out_channels, D_out, H_out, W_out),
                        dtype=x.dtype, device=x.device)

    # Pre-compute reciprocal of divisor for a multiplication instead of a division
    inv_divisor = 1.0 / divisor

    # Launch the fused kernel
    fused_ext.fused_conv_transpose3d_forward(
        x, weight, bias, output,
        N, C_in, out_channels,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        K, conv_transpose_stride, conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_groups, conv_transpose_dilation,
        min_value, inv_divisor)

    return output
