# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_070827/code_3.py
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
# CUDA source – fused convolution + bias kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused kernel: computes one output element (convolution + optional bias)
__global__ void conv_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const long long N,
    const long long C_in,
    const long long C_out,
    const long long H,
    const long long W,
    const long long K_h,
    const long long K_w,
    const long long OH,
    const long long OW,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int dilation_h,
    const int dilation_w,
    const int groups)
{
    const long long total = N * C_out * OH * OW;
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decode flat index -> (n, oc, oh, ow)
    long long rem = idx;
    const long long c_out_oh_ow = C_out * OH * OW;
    const long long n = rem / c_out_oh_ow;
    rem %= c_out_oh_ow;
    const long long oc = rem / (OH * OW);
    rem %= (OH * OW);
    const long long oh = rem / OW;
    const long long ow = rem % OW;

    // Group handling
    const long long c_out_per_group = C_out / groups;
    const long long c_in_per_group = C_in / groups;
    const long long g = oc / c_out_per_group;                // group id
    const long long ic_start = g * c_in_per_group;
    // (oc_start is not needed directly because weight is already organised by absolute oc)

    float sum = 0.0f;

    // Loop over kernel spatial positions
    for (long long ky = 0; ky < K_h; ++ky) {
        long long iy = oh * stride_h + ky * dilation_h - padding_h;
        if (iy < 0 || iy >= H) continue;
        for (long long kx = 0; kx < K_w; ++kx) {
            long long ix = ow * stride_w + kx * dilation_w - padding_w;
            if (ix < 0 || ix >= W) continue;

            // Loop over input channels belonging to this group
            for (long long ic = 0; ic < c_in_per_group; ++ic) {
                const long long actual_ic = ic_start + ic;

                // Input value
                const float v = input[(n * C_in + actual_ic) * H * W + iy * W + ix];

                // Weight value – weight layout is (C_out, C_in, K_h, K_w)
                const long long w_idx = ((oc * C_in) + actual_ic) * (K_h * K_w) + (ky * K_w + kx);
                const float w = weight[w_idx];

                sum += w * v;
            }
        }
    }

    // Add bias if present
    if (bias != nullptr) {
        sum += bias[oc];
    }

    // Store result
    output[(n * C_out + oc) * OH * OW + oh * OW + ow] = sum;
}

// Host-side wrapper that launches the kernel
void conv_bias_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    const long long N,
    const long long C_in,
    const long long C_out,
    const long long H,
    const long long W,
    const long long K_h,
    const long long K_w,
    const long long OH,
    const long long OW,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int dilation_h,
    const int dilation_w,
    const int groups)
{
    const long long total = N * C_out * OH * OW;
    const int block_dim = 256;                     // multiple of 32, good occupancy
    const int grid_dim = (total + block_dim - 1) / block_dim;

    const float* in_ptr   = input.data_ptr<float>();
    const float* w_ptr    = weight.data_ptr<float>();
    const float* b_ptr    = bias.defined() ? bias.data_ptr<float>() : nullptr;
    float* out_ptr        = output.data_ptr<float>();

    conv_bias_kernel<<<grid_dim, block_dim>>>(
        in_ptr, w_ptr, b_ptr, out_ptr,
        N, C_in, C_out, H, W, K_h, K_w, OH, OW,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);
    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
# C++ bindings (PyBind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void conv_bias_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    const long long N,
    const long long C_in,
    const long long C_out,
    const long long H,
    const long long W,
    const long long K_h,
    const long long K_w,
    const long long OH,
    const long long OW,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int dilation_h,
    const int dilation_w,
    const int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_bias_forward", &conv_bias_forward,
          "Fused convolution + bias forward kernel");
}
"""

# ----------------------------------------------------------------------
# Compile the fused-conv extension
# ----------------------------------------------------------------------
fused_conv_ext = load_inline(
    name='fused_conv',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# ----------------------------------------------------------------------
# functional_model – replaces the original F.conv2d call
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
    """
    Fused convolution (Conv2d) + optional bias.
    All arguments match the original signature except that we now run a custom
    CUDA kernel instead of PyTorch's built-in conv2d.
    """
    # Make sure tensors are contiguous (required for correct pointer arithmetic)
    device = x.device
    x = x.contiguous()
    weight = conv1d_weight.contiguous()

    # If no bias is supplied we pass an empty tensor; the kernel checks for a
    # nullptr and skips the bias addition.
    if conv1d_bias is not None:
        bias = conv1d_bias.contiguous()
    else:
        bias = torch.empty(0, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # Extract sizes
    # ------------------------------------------------------------------
    N, C_in, H, W = x.shape               # input shape
    C_out = weight.shape[0]                # output channels
    K_h = weight.shape[2]                  # kernel height
    K_w = weight.shape[3]                  # kernel width

    # ------------------------------------------------------------------
    # Helper to unpack possibly-tuple stride / padding / dilation
    # ------------------------------------------------------------------
    def _unpack(v, name):
        if isinstance(v, (tuple, list)):
            return int(v[0]), int(v[1]) if len(v) > 1 else int(v[0])
        return int(v), int(v)

    stride_h, stride_w = _unpack(conv1d_stride, "stride")
    pad_h,   pad_w     = _unpack(conv1d_padding, "padding")
    dil_h,   dil_w     = _unpack(conv1d_dilation, "dilation")

    # Compute output spatial size
    OH = (H + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    OW = (W + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1

    # Allocate output tensor
    output = torch.empty((N, C_out, OH, OW), dtype=x.dtype, device=device)

    # ------------------------------------------------------------------
    # Launch the fused kernel
    # ------------------------------------------------------------------
    fused_conv_ext.conv_bias_forward(
        x, weight, bias, output,
        N, C_in, C_out, H, W, K_h, K_w, OH, OW,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, int(conv1d_groups)
    )

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
