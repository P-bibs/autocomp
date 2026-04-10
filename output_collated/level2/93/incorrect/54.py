# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160018/code_6.py
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
# CUDA source – the fused transposed‑convolution + activation kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float fast_gelu(float x) {
    // Approximation used by many hardware vendors
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void conv_transpose_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const uint64_t bias_ptr,                 // 0 == nullptr
    float* __restrict__ output,
    const float add_val,
    const float mul_val,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int H_out, const int W_out,
    const int kH, const int kW,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int out_pad_h, const int out_pad_w,
    const int dilation_h, const int dilation_w,
    const int groups)
{
    const float* bias = bias_ptr ? reinterpret_cast<const float*>(bias_ptr) : nullptr;

    const int total = N * C_out * H_out * W_out;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // ---- decode flat index to (n, oc, oh, ow) ----
    int tmp = idx;
    const int n = tmp / (C_out * H_out * W_out);
    tmp        %= (C_out * H_out * W_out);
    const int oc = tmp / (H_out * W_out);
    tmp        %= (H_out * W_out);
    const int oh = tmp / W_out;
    const int ow = tmp % W_out;

    // ---- group information ----
    const int C_out_per_group = C_out / groups;
    const int group      = oc / C_out_per_group;
    const int oc_idx     = oc % C_out_per_group;
    const int C_in_per_group = C_in / groups;
    const int ic_start   = group * C_in_per_group;
    const int ic_end     = ic_start + C_in_per_group;

    float sum = 0.0f;

    // ---- convolution: accumulate over input channels and kernel ----
    for (int ic = ic_start; ic < ic_end; ++ic) {
        // base offset for this (ic, oc_idx) in the weight tensor
        const int w_base = ((group * C_out_per_group + oc_idx) * C_in_per_group + (ic - ic_start)) * kH * kW;
        // unroll the small 4×4 kernel manually
        #pragma unroll
        for (int ky = 0; ky < 4; ++ky) {
            const int i_h = oh * stride_h - pad_h + ky * dilation_h;
            if (i_h < 0 || i_h >= H_in) continue;
            #pragma unroll
            for (int kx = 0; kx < 4; ++kx) {
                const int i_w = ow * stride_w - pad_w + kx * dilation_w;
                if (i_w < 0 || i_w >= W_in) continue;
                const int in_idx = ((n * C_in + ic) * H_in + i_h) * W_in + i_w;
                const int w_idx  = w_base + (ky * kW + kx);
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }

    // ---- add bias ----
    if (bias) sum += bias[oc];

    // ---- fused activation: add → clamp → GELU → mul ----
    sum = sum + add_val;
    sum = fminf(sum, 0.0f);          // ReLU‑like clamp
    sum = fast_gelu(sum);
    sum = sum * mul_val;

    // ---- write result ----
    const int out_idx = ((n * C_out + oc) * H_out + oh) * W_out + ow;
    output[out_idx] = sum;
}

// ------------------------------------------------------------------
// Host‑side launcher
// ------------------------------------------------------------------
void conv_transpose_fused(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,           // may be empty (no bias)
    torch::Tensor output,
    float add_val,
    float mul_val,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w,
    int dilation_h, int dilation_w,
    int groups)
{
    const int N      = input.size(0);
    const int C_in   = input.size(1);
    const int H_in   = input.size(2);
    const int W_in   = input.size(3);

    const int C_out_per_group = weight.size(1);
    const int C_out = C_out_per_group * groups;   // out_channels_per_group * groups
    const int kH = weight.size(2);
    const int kW = weight.size(3);

    // output dimensions (already allocated by the caller)
    const int H_out = output.size(2);
    const int W_out = output.size(3);

    const int total = N * C_out * H_out * W_out;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    uint64_t bias_ptr = 0;
    if (bias.numel() > 0) bias_ptr = reinterpret_cast<uint64_t>(bias.data_ptr<float>());

    conv_transpose_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        add_val, mul_val,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kH, kW,
        stride_h, stride_w,
        pad_h, pad_w,
        out_pad_h, out_pad_w,
        dilation_h, dilation_w,
        groups);
}
"""

# ----------------------------------------------------------------------
# C++ binding (PYBIND11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void conv_transpose_fused(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float add_val,
    float mul_val,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w,
    int dilation_h, int dilation_w,
    int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose_fused", &conv_transpose_fused,
          "Fused transposed‑convolution + add‑min‑GELU‑mul");
}
"""

# ----------------------------------------------------------------------
# Compile the custom extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='conv_transpose_fused',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# ----------------------------------------------------------------------
# Model that uses the custom kernel
# ----------------------------------------------------------------------
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
    add_value,
    multiply_value,
):
    # ------------------------------------------------------------------
    # Compute output spatial size from the transposed‑convolution formula
    # ------------------------------------------------------------------
    stride = conv_transpose_stride          # integer, same for H and W
    pad    = conv_transpose_padding
    out_pad = conv_transpose_output_padding
    dil    = conv_transpose_dilation
    kH = conv_transpose_weight.size(2)
    kW = conv_transpose_weight.size(3)

    H_out = (x.size(2) - 1) * stride - 2 * pad + dil * (kH - 1) + out_pad + 1
    W_out = (x.size(3) - 1) * stride - 2 * pad + dil * (kW - 1) + out_pad + 1

    out_channels = conv_transpose_weight.size(1) * conv_transpose_groups

    # Allocate output tensor
    out = torch.empty(
        (x.size(0), out_channels, H_out, W_out),
        dtype=torch.float32,
        device=x.device,
    )

    # ------------------------------------------------------------------
    # Launch the fused CUDA kernel
    # ------------------------------------------------------------------
    fused_ext.conv_transpose_fused(
        x,
        conv_transpose_weight,
        conv_transpose_bias if conv_transpose_bias is not None else torch.empty(0, device='cuda'),
        out,
        float(add_value),
        float(multiply_value),
        stride, stride,            # stride_h, stride_w
        pad, pad,                  # pad_h, pad_w
        out_pad, out_pad,          # out_pad_h, out_pad_w
        dil, dil,                  # dilation_h, dilation_w
        conv_transpose_groups,
    )
    return out


# ----------------------------------------------------------------------
# Helper code – same constants as the original script
# ----------------------------------------------------------------------
batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
