# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_084802/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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
    # State for conv (nn.Conv2d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
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
# CUDA source – a single fused kernel that performs:
#   1) convolution (with arbitrary stride, padding, dilation, groups=1)
#   2) per‑pixel min across output channels
#   3) two tanh applications
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ input,   // (N, C_in, H, W)
    const float* __restrict__ weight,  // (C_out, C_in, K, K)
    const float* __restrict__ bias,    // (C_out) – may be nullptr
    float* __restrict__ output,        // (N, 1, out_h, out_w)
    const int N, const int C_in, const int H, const int W,
    const int C_out, const int K,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const int out_h, const int out_w,
    const int groups)                  // groups == 1 for this implementation
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * out_h * out_w) return;

    int n = idx / (out_h * out_w);
    int rem = idx % (out_h * out_w);
    int oh = rem / out_w;
    int ow = rem % out_w;

    float min_val = 1e38f;                     // start with a very large value

    // ------------------------------------------------------------------
    // Iterate over all output channels, compute the convolution and keep
    // the running minimum.
    // ------------------------------------------------------------------
    for (int oc = 0; oc < C_out; ++oc) {
        float sum = (bias != nullptr) ? bias[oc] : 0.0f;

        // convolution over input channels and kernel positions
        for (int ic = 0; ic < C_in; ++ic) {
            for (int kh = 0; kh < K; ++kh) {
                int i_h = oh * stride_h - pad_h + kh * dilation_h;
                if (i_h < 0 || i_h >= H) continue;          // boundary check
                for (int kw = 0; kw < K; ++kw) {
                    int i_w = ow * stride_w - pad_w + kw * dilation_w;
                    if (i_w < 0 || i_w >= W) continue;      // boundary check

                    float in_val = input[((n * C_in + ic) * H + i_h) * W + i_w];
                    float w_val  = weight[((oc * C_in + ic) * K + kh) * K + kw];
                    sum += in_val * w_val;
                }
            }
        }

        if (oc == 0 || sum < min_val) min_val = sum;
    }

    // ------------------------------------------------------------------
    // Apply tanh twice – using the fastmath version for speed.
    // ------------------------------------------------------------------
    float result = tanhf(min_val);
    result = tanhf(result);

    // Write the final scalar (channel dimension = 1)
    output[(n * out_h + oh) * out_w + ow] = result;
}

void fused_conv_min_tanh_launch(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int N, const int C_in, const int H, const int W,
    const int C_out, const int K,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const int out_h, const int out_w,
    const int groups)
{
    const int threads = 256;
    const int blocks = (N * out_h * out_w + threads - 1) / threads;

    fused_conv_min_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H, W,
        C_out, K,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        out_h, out_w,
        groups);
}
"""

# ----------------------------------------------------------------------
# C++ binding – exposes the launcher to Python
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_min_tanh_launch(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int N, const int C_in, const int H, const int W,
    const int C_out, const int K,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const int out_h, const int out_w,
    const int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_tanh", &fused_conv_min_tanh_launch,
          "Fused convolution → min → tanh·2");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# The functional model required by the evaluation harness
# ----------------------------------------------------------------------
def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
):
    """
    Fuses the convolution, the channel‑wise min reduction and the two
    tanh applications into a single custom CUDA kernel.
    """
    # ----------------------------------------------------------
    # Extract tensor shapes and convolution parameters
    # ----------------------------------------------------------
    N = x.size(0)          # batch size
    C_in = x.size(1)       # input channels
    H = x.size(2)          # input height
    W = x.size(3)          # input width

    C_out = conv_weight.size(0)   # output channels
    K = conv_weight.size(2)       # square kernel size

    # Normalise stride / padding / dilation to per‑dimension values
    if isinstance(conv_stride, int):
        stride_h = stride_w = conv_stride
    else:
        stride_h, stride_w = conv_stride

    if isinstance(conv_padding, int):
        pad_h = pad_w = conv_padding
    else:
        pad_h, pad_w = conv_padding

    if isinstance(conv_dilation, int):
        dilation_h = dilation_w = conv_dilation
    else:
        dilation_h, dilation_w = conv_dilation

    # ----------------------------------------------------------
    # Compute output spatial dimensions
    # ----------------------------------------------------------
    out_h = (H + 2 * pad_h - dilation_h * (K - 1) - 1) // stride_h + 1
    out_w = (W + 2 * pad_w - dilation_w * (K - 1) - 1) // stride_w + 1

    # ----------------------------------------------------------
    # Prepare output tensor (single channel after the min)
    # ----------------------------------------------------------
    output = torch.empty((N, 1, out_h, out_w), dtype=x.dtype, device=x.device)

    # If no bias has been supplied we simply use a zero tensor – adding
    # zero does not affect the result but lets us reuse the same kernel.
    if conv_bias is None:
        conv_bias = torch.zeros(C_out, dtype=x.dtype, device=x.device)

    # ----------------------------------------------------------
    # Launch the fused kernel
    # ----------------------------------------------------------
    fused_ext.fused_conv_min_tanh(
        x, conv_weight, conv_bias, output,
        N, C_in, H, W,
        C_out, K,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        out_h, out_w,
        conv_groups)

    return output

# ----------------------------------------------------------------------
# Helper functions required by the evaluation harness (not used internally)
# ----------------------------------------------------------------------
def get_init_inputs():
    return [16, 64, 3]   # in_channels, out_channels, kernel_size

def get_inputs():
    return [torch.rand(128, 16, 256, 256)]

# ----------------------------------------------------------------------
# Quick sanity check (can be removed – only for debugging)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(128, 16, 256, 256)
    w = torch.rand(64, 16, 3, 3)
    b = torch.rand(64)
    y = functional_model(x, conv_weight=w, conv_bias=b,
                         conv_stride=1, conv_padding=1,
                         conv_dilation=1, conv_groups=1)
    print("Output shape:", y.shape)   # (128, 1, 256, 256)
