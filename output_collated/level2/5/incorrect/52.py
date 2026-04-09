# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_121229/code_13.py
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

# -------------------------------------------------------------------------
# CUDA source – fused transposed convolution + bias subtraction + tanh
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose_tanh_kernel(
    const float* __restrict__ input,       // (N, C_in, H_in, W_in)
    const float* __restrict__ weight,      // (C_in, C_out_per_group, KH, KW)
    const float* __restrict__ conv_bias,   // (C_out) – may be nullptr
    const float* __restrict__ final_bias,  // (C_out)
    float* __restrict__ output,            // (N, C_out, H_out, W_out)
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int KH, const int KW,
    const int stride_h, const int stride_w,
    const int padding_h, const int padding_w,
    const int /*out_pad_h*/, const int /*out_pad_w*/, // not used in the inner loop
    const int groups,
    const int dilation_h, const int dilation_w,
    const int H_out, const int W_out)
{
    const long long total_out = (long long)N * C_out * H_out * W_out;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    // decode (n, c, oh, ow)
    const int n = idx / (C_out * H_out * W_out);
    int rem = idx % (C_out * H_out * W_out);
    const int c = rem / (H_out * W_out);
    rem = rem % (H_out * W_out);
    const int oh = rem / W_out;
    const int ow = rem % W_out;

    // group information
    const int out_ch_per_group = C_out / groups;
    const int in_ch_per_group = C_in / groups;
    const int g = c / out_ch_per_group;
    const int oc_local = c - g * out_ch_per_group;

    float sum = 0.0f;

    // loop over input channels belonging to the same group
    for (int ic = g * in_ch_per_group; ic < (g + 1) * in_ch_per_group; ++ic) {
        const int ic_local = ic - g * in_ch_per_group;
        // loop over kernel
        for (int kh = 0; kh < KH; ++kh) {
            int i_h = oh + padding_h - dilation_h * kh;
            if (i_h < 0) continue;
            const int ih = i_h / stride_h;
            if (ih * stride_h != i_h) continue;
            if (ih >= H_in) continue;

            for (int kw = 0; kw < KW; ++kw) {
                int i_w = ow + padding_w - dilation_w * kw;
                if (i_w < 0) continue;
                const int iw = i_w / stride_w;
                if (iw * stride_w != i_w) continue;
                if (iw >= W_in) continue;

                // input value
                const int in_idx = ((n * C_in + ic) * H_in + ih) * W_in + iw;
                const float inp_val = input[in_idx];

                // weight value
                const int w_idx = ((ic_local * out_ch_per_group + oc_local) * KH + kh) * KW + kw;
                const float w_val = weight[w_idx];

                // accumulate
                sum += inp_val * w_val;
            }
        }
    }

    // optional convolution bias
    if (conv_bias != nullptr) {
        sum += conv_bias[c];
    }

    // user‑supplied bias (subtracted before tanh)
    sum -= final_bias[c];

    // tanh – use the fast version
    sum = tanhf(sum);

    // store result
    const int out_idx = ((n * C_out + c) * H_out + oh) * W_out + ow;
    output[out_idx] = sum;
}

void fused_conv_transpose_tanh(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> conv_bias,
    torch::Tensor final_bias,
    torch::Tensor output,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int groups,
    int dilation_h, int dilation_w,
    int H_out, int W_out)
{
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    const int C_out = weight.size(1) * groups;
    const int KH = weight.size(2);
    const int KW = weight.size(3);

    const long long total = (long long)N * C_out * H_out * W_out;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65536) blocks = 65536;

    const float* conv_bias_ptr = conv_bias.has_value() ? conv_bias->data_ptr<float>() : nullptr;

    fused_conv_transpose_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias_ptr,
        final_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, KH, KW,
        stride_h, stride_w,
        padding_h, padding_w,
        output_padding_h, output_padding_w,
        groups,
        dilation_h, dilation_w,
        H_out, W_out);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");
}
"""

# -------------------------------------------------------------------------
# Host (C++) code – pybind11 binding
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
#include <c10/util/Optional.h>

void fused_conv_transpose_tanh(
    torch::Tensor input,                     // (N, C_in, H_in, W_in)
    torch::Tensor weight,                    // (C_in, C_out/groups, KH, KW)
    c10::optional<torch::Tensor> conv_bias,  // optional per‑channel bias after convolution
    torch::Tensor final_bias,                // per-channel bias to subtract before tanh
    torch::Tensor output,                    // (N, C_out, H_out, W_out) – pre‑allocated
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int groups,
    int dilation_h, int dilation_w,
    int H_out, int W_out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_tanh",
          &fused_conv_transpose_tanh,
          "Fused transposed convolution + bias subtraction + tanh");
}
"""

# -------------------------------------------------------------------------
# Compile the inline extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# -------------------------------------------------------------------------
# Helper to compute output spatial size of a transposed convolution
# -------------------------------------------------------------------------
def conv_transpose_output_shape(x: torch.Tensor, weight: torch.Tensor,
                                 stride, padding, output_padding,
                                 groups, dilation):
    """
    Returns (C_out, H_out, W_out) for a torch.nn.functional.conv_transpose2d
    with the given parameters.
    """
    N = x.size(0)
    C_in = x.size(1)
    H_in = x.size(2)
    W_in = x.size(3)

    C_out = weight.size(1) * groups
    KH = weight.size(2)
    KW = weight.size(3)

    H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (KH - 1) + output_padding[0] + 1
    W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (KW - 1) + output_padding[1] + 1
    return C_out, H_out, W_out

# -------------------------------------------------------------------------
# Functional model – replaces the original conv_transpose2d + custom kernel
# -------------------------------------------------------------------------
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
    bias,
):
    """
    Fused transposed convolution, bias subtraction and tanh.
    All parameters are the same as in the original `functional_model`,
    but we no longer call `torch.conv_transpose2d`.
    """
    # -----------------------------------------------------------------
    # 1. Determine output shape
    # -----------------------------------------------------------------
    C_out, H_out, W_out = conv_transpose_output_shape(
        x, conv_transpose_weight,
        stride=conv_transpose_stride,
        padding=conv_transpose_padding,
        output_padding=conv_transpose_output_padding,
        groups=conv_transpose_groups,
        dilation=conv_transpose_dilation
    )

    # -----------------------------------------------------------------
    # 2. Allocate output tensor
    # -----------------------------------------------------------------
    output = torch.empty((x.size(0), C_out, H_out, W_out),
                         dtype=torch.float32, device='cuda')

    # -----------------------------------------------------------------
    # 3. Flatten the biases (they are 1‑D vectors of length C_out)
    # -----------------------------------------------------------------
    final_bias = bias.view(-1).contiguous()               # (C_out,)

    if conv_transpose_bias is not None:
        conv_bias = conv_transpose_bias.view(-1).contiguous()   # (C_out,)
    else:
        conv_bias = None                                       # will be passed as nullptr

    # -----------------------------------------------------------------
    # 4. Launch the fused kernel
    # -----------------------------------------------------------------
    fused_ext.fused_conv_transpose_tanh(
        x,                      # input tensor (N, C_in, H_in, W_in)
        conv_transpose_weight,  # weight tensor (C_in, C_out/groups, KH, KW)
        conv_bias,              # optional convolution bias
        final_bias,             # bias to subtract before tanh
        output,                 # pre‑allocated output tensor
        conv_transpose_stride[0],   # stride_h
        conv_transpose_stride[1],   # stride_w
        conv_transpose_padding[0], # padding_h
        conv_transpose_padding[1], # padding_w
        conv_transpose_output_padding[0], # output_padding_h
        conv_transpose_output_padding[1], # output_padding_w
        conv_transpose_groups,
        conv_transpose_dilation[0],  # dilation_h
        conv_transpose_dilation[1],  # dilation_w
        H_out,
        W_out
    )

    return output
