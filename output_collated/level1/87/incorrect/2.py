# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_065229/code_3.py
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

# ----------------------------------------------------------------------
# CUDA source – tiled 2‑D convolution kernel + host wrapper
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int TILE_H = 16;
constexpr int TILE_W = 16;

__global__ void conv2d_fwd_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_ch,
    const int out_ch,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups)
{
    const int tile_w_idx = blockIdx.x;
    const int tile_h_idx = blockIdx.y;
    const int bc = blockIdx.z;

    const int b   = bc / out_ch;
    const int oc  = bc % out_ch;

    const int oh_base = tile_h_idx * TILE_H;
    const int ow_base = tile_w_idx * TILE_W;

    const int tid_y = threadIdx.y;
    const int tid_x = threadIdx.x;

    const int oh = oh_base + tid_y;
    const int ow = ow_base + tid_x;

    if (oh >= out_h || ow >= out_w) return;

    // ----- group decomposition -----
    const int in_ch_per_group  = in_ch / groups;
    const int out_ch_per_group = out_ch / groups;
    const int g                = oc / out_ch_per_group;
    const int oc_group         = oc - g * out_ch_per_group;

    // ----- base offsets -----
    const int input_batch_offset = b * in_ch * in_h * in_w;
    const int input_group_offset = g * in_ch_per_group * in_h * in_w;
    const int weight_group_offset = (g * out_ch_per_group + oc_group) *
                                    (in_ch_per_group * kernel_h * kernel_w);

    float sum = 0.0f;

    // ----- loop over input channels in this group -----
    for (int ic = 0; ic < in_ch_per_group; ++ic) {
        const int weight_ic_offset = weight_group_offset + ic * (kernel_h * kernel_w);
        for (int ky = 0; ky < kernel_h; ++ky) {
            const int iy = oh * stride_h - pad_h + ky * dilation_h;
            if (iy < 0 || iy >= in_h) continue;
            const int input_row_offset = input_batch_offset + input_group_offset +
                                         ic * (in_h * in_w) + iy * in_w;
            for (int kx = 0; kx < kernel_w; ++kx) {
                const int ix = ow * stride_w - pad_w + kx * dilation_w;
                if (ix < 0 || ix >= in_w) continue;
                const float inp_val = input[input_row_offset + ix];
                const float w_val   = weight[weight_ic_offset + ky * kernel_w + kx];
                sum += inp_val * w_val;
            }
        }
    }

    if (bias != nullptr) sum += bias[oc];

    // ----- write result (NCHW layout) -----
    const size_t out_idx = ((size_t)b * out_ch + oc) * out_h + oh;
    const size_t out_idx_final = out_idx * out_w + ow;
    output[out_idx_final] = sum;
}

void fused_conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch,
    int in_ch,
    int out_ch,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups)
{
    const float* input_ptr  = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr   = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;
    float*       output_ptr = output.data_ptr<float>();

    dim3 block(TILE_W, TILE_H, 1);
    dim3 grid( (out_w + TILE_W - 1) / TILE_W,
               (out_h + TILE_H - 1) / TILE_H,
               batch * out_ch );

    conv2d_fwd_kernel<<<grid, block>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        batch, in_ch, out_ch,
        in_h, in_w,
        out_h, out_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        groups);

    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
# C++ binding – exposes the kernel to Python
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch,
    int in_ch,
    int out_ch,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv2d", &fused_conv2d, "Fused Conv2D forward");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_conv',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True)

# ----------------------------------------------------------------------
# Parameters used in the benchmark (identical to the original file)
# ----------------------------------------------------------------------
batch_size    = 16
in_channels   = 64
out_channels  = 128
height = width = 1024

TILE_H = 16
TILE_W = 16

# ----------------------------------------------------------------------
# Helper that launches the tiled kernel
# ----------------------------------------------------------------------
def conv2d_fwd(x, weight, bias=None,
               stride=1, padding=0, dilation=1, groups=1):
    """Own CUDA implementation of 2‑D convolution (tiled)."""
    device = x.device

    # Make sure data is contiguous and on the GPU
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    else:
        # Dummy empty tensor – will be interpreted as a null pointer
        bias = torch.empty(0, dtype=x.dtype, device=device)

    batch, in_ch, in_h, in_w = x.shape
    out_ch = weight.shape[0]
    kH = weight.shape[2]
    kW = weight.shape[3]

    # Normalise stride / padding / dilation to integer pairs
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    if isinstance(dilation, int):
        dil_h = dil_w = dilation
    else:
        dil_h, dil_w = dilation

    out_h = (in_h + 2*pad_h - dil_h*(kH-1) - 1)//stride_h + 1
    out_w = (in_w + 2*pad_w - dil_w*(kW-1) - 1)//stride_w + 1

    output = torch.empty((batch, out_ch, out_h, out_w),
                         dtype=x.dtype, device=device)

    # Grid dimensions – one block per 16×16 output tile
    grid_x = (out_w + TILE_W - 1) // TILE_W
    grid_y = (out_h + TILE_H - 1) // TILE_H
    grid_z = batch * out_ch

    fused_ext.fused_conv2d(
        x, weight, bias, output,
        batch, in_ch, out_ch, in_h, in_w,
        out_h, out_w,
        kH, kW,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        groups)

    return output

# ----------------------------------------------------------------------
# The only function that will be imported by the evaluator
# ----------------------------------------------------------------------
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
    """Replaces the original F.conv2d call with the tiled CUDA kernel."""
    return conv2d_fwd(
        x,
        conv1d_weight,
        conv1d_bias,
        stride=conv1d_stride,
        padding=conv1d_padding,
        dilation=conv1d_dilation,
        groups=conv1d_groups)

# ----------------------------------------------------------------------
# Optional helpers (kept for completeness)
# ----------------------------------------------------------------------
def get_init_inputs():
    return [in_channels, out_channels]

def get_inputs():
    # Input tensor allocated on the GPU as expected by the evaluator
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
