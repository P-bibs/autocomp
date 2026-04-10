# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_071708/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv1d_weight', 'conv1d_bias', 'conv1d_stride', 'conv1d_padding', 'conv1d_dilation', 'conv1d_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv1d_weight', 'conv1d_bias']


class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool=False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

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
    # State for conv1d (nn.Conv2d)
    if 'conv1d_weight' in flat_state:
        state_kwargs['conv1d_weight'] = flat_state['conv1d_weight']
    else:
        state_kwargs['conv1d_weight'] = getattr(model.conv1d, 'weight', None)
    if 'conv1d_bias' in flat_state:
        state_kwargs['conv1d_bias'] = flat_state['conv1d_bias']
    else:
        state_kwargs['conv1d_bias'] = getattr(model.conv1d, 'bias', None)
    state_kwargs['conv1d_stride'] = model.conv1d.stride
    state_kwargs['conv1d_padding'] = model.conv1d.padding
    state_kwargs['conv1d_dilation'] = model.conv1d.dilation
    state_kwargs['conv1d_groups'] = model.conv1d.groups
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
# CUDA source – the convolution kernel and a host-side wrapper
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

// ---------------------------------------------------------------------
// Naïve 2-D convolution kernel
// ---------------------------------------------------------------------
__global__ void conv2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int64_t total_out,
    const int N, const int C, const int H, const int W,
    const int OC, const int in_ch_per_g, const int G,
    const int K_H, const int K_W,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const int out_h, const int out_w,
    float* __restrict__ out)
{
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total_out; i += stride) {
        // ---- decode output index (n, oc, oh, ow) ----
        const int64_t total_out_per_img = (int64_t)OC * out_h * out_w;
        const int n      = i / total_out_per_img;
        const int rem0   = i % total_out_per_img;
        const int oc     = rem0 / (out_h * out_w);
        const int rem1   = rem0 % (out_h * out_w);
        const int oh     = rem1 / out_w;
        const int ow     = rem1 % out_w;

        // ---- initialise with bias (if any) ----
        float sum = (bias != nullptr) ? bias[oc] : 0.0f;

        // ---- convolution per group ----
        const int oc_per_group = OC / G;
        const int group_id     = oc / oc_per_group;
        const int oc_local     = oc % oc_per_group;
        const int ic_start     = group_id * in_ch_per_g;

        // loop over input channels inside the group
        for (int ic = 0; ic < in_ch_per_g; ++ic) {
            const int actual_ic = ic_start + ic;

            // loop over kernel spatial positions
            for (int kh = 0; kh < K_H; ++kh) {
                const int ih = oh * stride_h - pad_h + kh * dilation_h;
                if (ih < 0 || ih >= H) continue;

                for (int kw = 0; kw < K_W; ++kw) {
                    const int iw = ow * stride_w - pad_w + kw * dilation_w;
                    if (iw < 0 || iw >= W) continue;

                    // ---- indices into flattened arrays ----
                    const int64_t x_idx = (((int64_t)n * C + actual_ic) * H + ih) * W + iw;
                    const int weight_idx = ((oc_local * in_ch_per_g + ic) * K_H + kh) * K_W + kw;

                    const float xv = x[x_idx];
                    const float wv = weight[weight_idx];
                    sum += xv * wv;
                }
            }
        }
        out[i] = sum;
    }
}

// ---------------------------------------------------------------------
// Host-side wrapper callable from Python
// ---------------------------------------------------------------------
void fused_conv2d(
    at::Tensor x,
    at::Tensor weight,
    at::Tensor bias,
    int64_t total_out,
    int N, int C, int H, int W,
    int OC, int in_ch_per_g, int G,
    int K_H, int K_W,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int out_h, int out_w,
    at::Tensor output)
{
    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    float* o_ptr = output.data_ptr<float>();

    const int block_size = 256;
    const int grid_size  = std::min<int>((total_out + block_size - 1) / block_size, 65535);

    conv2d_kernel<<<grid_size, block_size>>>(
        x_ptr, w_ptr, b_ptr,
        total_out,
        N, C, H, W,
        OC, in_ch_per_g, G,
        K_H, K_W,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        out_h, out_w,
        o_ptr);

    cudaDeviceSynchronize();
}
"""

# -------------------------------------------------------------------------
# C++ binding – exposes the wrapper to Python via pybind11
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_conv2d(
    at::Tensor x,
    at::Tensor weight,
    at::Tensor bias,
    int64_t total_out,
    int N, int C, int H, int W,
    int OC, int in_ch_per_g, int G,
    int K_H, int K_W,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int out_h, int out_w,
    at::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv2d", &fused_conv2d,
          "Custom CUDA 2-D convolution (forward)");
}
"""

# -------------------------------------------------------------------------
# Build the inline CUDA extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_conv2d',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# -------------------------------------------------------------------------
# Helper to turn a single int into a (h,w) tuple
# -------------------------------------------------------------------------
def _to_tuple(val, dim=2):
    if isinstance(val, int):
        return (val,) * dim
    return val

# -------------------------------------------------------------------------
# The optimized functional_model – replaces PyTorch's F.conv2d with the
# hand-written CUDA kernel
# -------------------------------------------------------------------------
def functional_model(
    x,
    *,
    conv1d_weight,
    conv1d_bias,
    conv1d_stride,
    conv1d_padding,
    conv1d_dilation,
    conv1d_groups,
):
    # -----------------------------------------------------------------
    # Move tensors to the GPU and make them contiguous
    # -----------------------------------------------------------------
    x = x.cuda().contiguous()
    weight = conv1d_weight.cuda().contiguous()
    bias = conv1d_bias.cuda().contiguous() if conv1d_bias is not None else torch.empty(0, device='cuda')

    N, C, H, W = x.shape
    OC = weight.shape[0]                # output channels
    in_ch_per_g = weight.shape[1]       # input channels per group
    G = conv1d_groups
    K_H = weight.shape[2]
    K_W = weight.shape[3]

    # -----------------------------------------------------------------
    # Compute output spatial size (same formula as PyTorch)
    # -----------------------------------------------------------------
    stride = _to_tuple(conv1d_stride)
    pad    = _to_tuple(conv1d_padding)
    dil    = _to_tuple(conv1d_dilation)

    out_h = (H + 2 * pad[0] - dil[0] * (K_H - 1) - 1) // stride[0] + 1
    out_w = (W + 2 * pad[1] - dil[1] * (K_W - 1) - 1) // stride[1] + 1

    # -----------------------------------------------------------------
    # Allocate output tensor on the GPU
    # -----------------------------------------------------------------
    output = torch.empty((N, OC, out_h, out_w), dtype=x.dtype, device='cuda')

    # -----------------------------------------------------------------
    # Total number of output elements – used for grid-stride loop
    # -----------------------------------------------------------------
    total_out = N * OC * out_h * out_w

    # -----------------------------------------------------------------
    # Call the custom CUDA kernel
    # -----------------------------------------------------------------
    fused_ext.fused_conv2d(
        x, weight, bias,
        total_out,
        N, C, H, W,
        OC, in_ch_per_g, G,
        K_H, K_W,
        stride[0], stride[1],
        pad[0], pad[1],
        dil[0], dil[1],
        out_h, out_w,
        output)

    return output

batch_size = 16
in_channels = 64
out_channels = 128
width = 1024
height = 1024

def get_init_inputs():
    return [in_channels, out_channels]

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]
