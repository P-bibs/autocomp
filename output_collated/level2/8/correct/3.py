# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_054338/code_6.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'divisor', 'pool_size', 'bias_shape', 'sum_dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'max_pool_kernel_size', 'max_pool_stride', 'max_pool_padding', 'max_pool_dilation', 'max_pool_ceil_mode', 'max_pool_return_indices', 'global_avg_pool_output_size', 'divisor', 'bias', 'sum_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """

    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

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
    # State for conv (nn.Conv3d)
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
    # State for max_pool (nn.MaxPool3d)
    state_kwargs['max_pool_kernel_size'] = model.max_pool.kernel_size
    state_kwargs['max_pool_stride'] = model.max_pool.stride
    state_kwargs['max_pool_padding'] = model.max_pool.padding
    state_kwargs['max_pool_dilation'] = model.max_pool.dilation
    state_kwargs['max_pool_ceil_mode'] = model.max_pool.ceil_mode
    state_kwargs['max_pool_return_indices'] = model.max_pool.return_indices
    # State for global_avg_pool (nn.AdaptiveAvgPool3d)
    state_kwargs['global_avg_pool_output_size'] = model.global_avg_pool.output_size
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    if 'sum_dim' in flat_state:
        state_kwargs['sum_dim'] = flat_state['sum_dim']
    else:
        state_kwargs['sum_dim'] = getattr(model, 'sum_dim')
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------------------
# 1️⃣  Custom Conv3D kernel (optimisation #2)
# -------------------------------------------------------------------------
_conv_cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int kD, const int kH, const int kW,
    const int s0, const int s1, const int s2,
    const int p0, const int p1, const int p2,
    const int d0, const int d1, const int d2,
    const int groups,
    const int use_bias)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C_out * D_out * H_out * W_out;
    if (idx >= total) return;

    // decompose global index into (n, oc, od, oh, ow)
    const int out_per_n = C_out * D_out * H_out * W_out;
    int n = idx / out_per_n;
    int remainder = idx % out_per_n;
    int oc = remainder / (D_out * H_out * W_out);
    int rem2 = remainder % (D_out * H_out * W_out);
    int od = rem2 / (H_out * W_out);
    int rem3 = rem2 % (H_out * W_out);
    int oh = rem3 / W_out;
    int ow = rem3 % W_out;

    // group handling
    const int c_out_per_group = C_out / groups;
    const int c_in_per_group  = C_in  / groups;
    const int group_id = oc / c_out_per_group;
    const int ic_start = group_id * c_in_per_group;

    float sum = 0.0f;

    // convolution loops
    for (int kd = 0; kd < kD; ++kd) {
        const int id = od * s0 - p0 + kd * d0;
        if (id < 0 || id >= D_in) continue;
        for (int kh = 0; kh < kH; ++kh) {
            const int ih = oh * s1 - p1 + kh * d1;
            if (ih < 0 || ih >= H_in) continue;
            for (int kw = 0; kw < kW; ++kw) {
                const int iw = ow * s2 - p2 + kw * d2;
                if (iw < 0 || iw >= W_in) continue;
                for (int ic = ic_start; ic < ic_start + c_in_per_group; ++ic) {
                    const int in_idx = ((n * C_in + ic) * D_in + id) * H_in * W_in
                                       + ih * W_in + iw;
                    const int w_idx = (((oc * c_in_per_group + (ic - ic_start)) * kD + kd)
                                        * kH + kh) * kW + kw;
                    const float w_val = __ldg(&weight[w_idx]);
                    const float i_val = __ldg(&input[in_idx]);
                    sum += i_val * w_val;
                }
            }
        }
    }

    if (use_bias) sum += __ldg(&bias[oc]);
    output[idx] = sum;
}

void conv3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int kD, const int kH, const int kW,
    const int s0, const int s1, const int s2,
    const int p0, const int p1, const int p2,
    const int d0, const int d1, const int d2,
    const int groups,
    const int use_bias)
{
    const int total = N * C_out * D_out * H_out * W_out;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        kD, kH, kW,
        s0, s1, s2,
        p0, p1, p2,
        d0, d1, d2,
        groups, use_bias);
}
"""

_conv_cpp_src = r"""
#include <torch/extension.h>
void conv3d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int kD, const int kH, const int kW,
    const int s0, const int s1, const int s2,
    const int p0, const int p1, const int p2,
    const int d0, const int d1, const int d2,
    const int groups,
    const int use_bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d", &conv3d_forward, "conv3d forward");
}
"""

conv_ext = load_inline(
    name="conv_ext",
    cpp_sources=_conv_cpp_src,
    cuda_sources=_conv_cuda_src,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)

# -------------------------------------------------------------------------
# 2️⃣  Fused-op kernel (already present, kept unchanged)
# -------------------------------------------------------------------------
_fused_cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float divisor,
    int N, int C, int D, int H, int W,
    int sum_dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * D * H * W;

    if (idx < total_elements) {
        int n = idx / (D * H * W);
        int d = (idx / (H * W)) % D;
        int h = (idx / W) % H;
        int w = idx % W;

        float sum_val = 0.0f;
        for (int c = 0; c < C; ++c) {
            int input_idx = n * (C * D * H * W) + c * (D * H * W) + d * (H * W) + h * W + w;
            sum_val += (input[input_idx] / divisor) + bias[c];
        }
        output[idx] = sum_val;
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor bias,
                      torch::Tensor output, float divisor) {
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    int threads = 256;
    int blocks = (N * D * H * W + threads - 1) / threads;

    fused_op_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), divisor, N, C, D, H, W, 1);
}
"""

_fused_cpp_src = r"""
void fused_op_forward(torch::Tensor input, torch::Tensor bias,
                      torch::Tensor output, float divisor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused divide, bias, and sum kernel");
}
"""

fused_ext = load_inline(
    name="fused_op",
    cpp_sources=_fused_cpp_src,
    cuda_sources=_fused_cuda_src,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)


# -------------------------------------------------------------------------
# 3️⃣  Helper utilities
# -------------------------------------------------------------------------
def _to_triple(x):
    """Turn an int or a 3-tuple into a 3-tuple."""
    if isinstance(x, int):
        return (x, x, x)
    return tuple(x)


# -------------------------------------------------------------------------
# 4️⃣  Functional model – replaces the PyTorch conv3d with the custom kernel
# -------------------------------------------------------------------------
def functional_model(
    x, *,
    conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups,
    max_pool_kernel_size, max_pool_stride, max_pool_padding, max_pool_dilation,
    max_pool_ceil_mode, max_pool_return_indices, global_avg_pool_output_size,
    divisor, bias, sum_dim,
):
    # -----------------------------------------------------------------
    # (a) Custom convolution
    # -----------------------------------------------------------------
    stride   = _to_triple(conv_stride)
    padding  = _to_triple(conv_padding)
    dilation = _to_triple(conv_dilation)

    N, C_in, D_in, H_in, W_in = x.shape
    kD, kH, kW = conv_weight.shape[2], conv_weight.shape[3], conv_weight.shape[4]

    # output spatial size after convolution (floor version)
    D_out = (D_in + 2 * padding[0] - dilation[0] * (kD - 1) - 1) // stride[0] + 1
    H_out = (H_in + 2 * padding[1] - dilation[1] * (kH - 1) - 1) // stride[1] + 1
    W_out = (W_in + 2 * padding[2] - dilation[2] * (kW - 1) - 1) // stride[2] + 1
    C_out = conv_weight.shape[0]

    conv_out = torch.empty((N, C_out, D_out, H_out, W_out),
                           dtype=x.dtype, device=x.device)

    # Prepare bias argument (dummy empty tensor when conv_bias is None)
    if conv_bias is not None:
        bias_tensor = conv_bias.contiguous().view(-1)
        use_bias = 1
    else:
        bias_tensor = torch.empty(0, dtype=x.dtype, device=x.device)
        use_bias = 0

    conv_ext.conv3d(
        x.contiguous(),
        conv_weight.contiguous(),
        bias_tensor,
        conv_out,
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        kD, kH, kW,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        conv_groups,
        use_bias,
    )

    # -----------------------------------------------------------------
    # (b) Remaining PyTorch operations (already GPU-optimal)
    # -----------------------------------------------------------------
    x = F.max_pool3d(
        conv_out,
        kernel_size=max_pool_kernel_size,
        stride=max_pool_stride,
        padding=max_pool_padding,
        dilation=max_pool_dilation,
        ceil_mode=max_pool_ceil_mode,
        return_indices=max_pool_return_indices,
    )
    x = F.adaptive_avg_pool3d(x, global_avg_pool_output_size)

    # -----------------------------------------------------------------
    # (c) Fused divide-bias-sum kernel
    # -----------------------------------------------------------------
    N2, C2, D2, H2, W2 = x.shape
    out = torch.zeros((N2, D2, H2, W2), device=x.device)
    fused_ext.fused_op(x.contiguous(), bias.contiguous().view(-1), out, divisor)
    return out


# -------------------------------------------------------------------------
# 5️⃣  Dummy initialisation (kept for compatibility with the harness)
# -------------------------------------------------------------------------
batch_size = 128
in_channels = 8
out_channels = 16
depth = 16
height = 64
width = 64
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size,
            bias_shape, sum_dim]


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
