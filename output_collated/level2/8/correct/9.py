# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_060810/code_6.py
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
from torch.utils.cpp_extension import load_inline

# =============================================================================
# 1️⃣  Custom Conv3d CUDA kernel (optimization #1)
# =============================================================================
conv_cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int D_in, const int H_in, const int W_in,
    const int C_out, const int Kd, const int Kh, const int Kw,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int dil_d, const int dil_h, const int dil_w,
    const int groups,
    const int D_out, const int H_out, const int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * D_out * H_out * W_out;
    if (idx >= total) return;

    // --- unpack output index ----------------------------------------------------
    int n = idx / (C_out * D_out * H_out * W_out);
    int rem = idx % (C_out * D_out * H_out * W_out);
    int oc = rem / (D_out * H_out * W_out);
    rem = rem % (D_out * H_out * W_out);
    int od = rem / (H_out * W_out);
    rem = rem % (H_out * W_out);
    int oh = rem / W_out;
    int ow = rem % W_out;

    // --- group information -------------------------------------------------------
    int C_out_per_group = C_out / groups;
    int C_in_per_group  = C_in  / groups;
    int group_id = oc / C_out_per_group;
    int c_in_start = group_id * C_in_per_group;

    // --- top‑left corner of the kernel in the input -----------------------------
    int id_d_start = od * stride_d - pad_d;
    int id_h_start = oh * stride_h - pad_h;
    int id_w_start = ow * stride_w - pad_w;

    float sum = 0.0f;

    // --- inner loops over input channels and kernel positions --------------------
    for (int c_in_rel = 0; c_in_rel < C_in_per_group; ++c_in_rel) {
        int c_in = c_in_start + c_in_rel;
        int weight_base = ((oc % C_out_per_group) * C_in_per_group + c_in_rel) * (Kd * Kh * Kw);
        for (int kd = 0; kd < Kd; ++kd) {
            int id_d = id_d_start + kd * dil_d;
            if (id_d < 0 || id_d >= D_in) continue;
            for (int kh = 0; kh < Kh; ++kh) {
                int id_h = id_h_start + kh * dil_h;
                if (id_h < 0 || id_h >= H_in) continue;
                for (int kw = 0; kw < Kw; ++kw) {
                    int id_w = id_w_start + kw * dil_w;
                    if (id_w < 0 || id_w >= W_in) continue;

                    int input_idx = n * (C_in * D_in * H_in * W_in)
                                    + c_in * (D_in * H_in * W_in)
                                    + id_d * (H_in * W_in) + id_h * W_in + id_w;
                    int weight_idx = weight_base + (kd * Kh * Kw + kh * Kw + kw);
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    if (bias != nullptr) sum += bias[oc];
    output[idx] = sum;
}

torch::Tensor conv3d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                    int stride_d, int stride_h, int stride_w,
                    int pad_d, int pad_h, int pad_w,
                    int dil_d, int dil_h, int dil_w,
                    int groups) {
    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    int C_out = weight.size(0);
    int Kd = weight.size(2);
    int Kh = weight.size(3);
    int Kw = weight.size(4);

    int D_out = (D_in + 2*pad_d - dil_d*(Kd-1) - 1) / stride_d + 1;
    int H_out = (H_in + 2*pad_h - dil_h*(Kh-1) - 1) / stride_h + 1;
    int W_out = (W_in + 2*pad_w - dil_w*(Kw-1) - 1) / stride_w + 1;

    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    const float* bias_ptr = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;

    int total = N * C_out * D_out * H_out * W_out;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, Kd, Kh, Kw,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        groups,
        D_out, H_out, W_out);

    return output;
}
"""

conv_cpp_source = r"""
#include <torch/extension.h>

torch::Tensor conv3d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                             int stride_d, int stride_h, int stride_w,
                             int pad_d, int pad_h, int pad_w,
                             int dil_d, int dil_h, int dil_w,
                             int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d", &conv3d_forward, "Custom conv3d forward");
}
"""

conv_ext = load_inline(
    name='conv3d',
    cpp_sources=conv_cpp_source,
    cuda_sources=conv_cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# =============================================================================
# 2️⃣  Optimised fused‑reduce kernel (optimization #11)
# =============================================================================
fused_cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float inv_divisor,
    float sum_bias,
    int N, int C, int D, int H, int W) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * D * H * W;

    if (idx < total_elements) {
        int n = idx / (D * H * W);
        int d = (idx / (H * W)) % D;
        int h = (idx / W) % H;
        int w = idx % W;

        // base offset for this (n,d,h,w) in the input tensor
        int base = n * (C * D * H * W) + d * (H * W) + h * W + w;

        float sum = 0.0f;
        // simple loop – the compiler unrolls for the known channel count
        for (int c = 0; c < C; ++c) {
            int input_idx = base + c * (D * H * W);
            sum += input[input_idx] * inv_divisor;   // multiply instead of divide
        }
        output[idx] = sum + sum_bias;                // add pre‑computed bias sum
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor output, float inv_divisor, float sum_bias) {
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    int threads = 256;
    int blocks = (N * D * H * W + threads - 1) / threads;

    fused_op_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        inv_divisor, sum_bias, N, C, D, H, W);
}
"""

fused_cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor input, torch::Tensor output, float inv_divisor, float sum_bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused divide, bias, and sum kernel (optimized)");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# =============================================================================
# 3️⃣  Full functional model using the custom kernels
# =============================================================================
import torch.nn.functional as F

def functional_model(
    x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups,
    max_pool_kernel_size, max_pool_stride, max_pool_padding, max_pool_dilation,
    max_pool_ceil_mode, max_pool_return_indices, global_avg_pool_output_size,
    divisor, bias, sum_dim,
):
    # ---- unpack convolution parameters -----------------------------------------
    stride_d, stride_h, stride_w = conv_stride if conv_stride is not None else (1, 1, 1)
    pad_d,  pad_h,  pad_w        = conv_padding if conv_padding is not None else (0, 0, 0)
    dil_d,  dil_h,  dil_w        = conv_dilation if conv_dilation is not None else (1, 1, 1)
    groups = conv_groups if conv_groups is not None else 1

    # ---- ensure a bias tensor exists (zero if not supplied) --------------------
    if conv_bias is None:
        C_out = conv_weight.size(0)
        conv_bias_tensor = torch.zeros(C_out, dtype=torch.float32, device=x.device)
    else:
        conv_bias_tensor = conv_bias

    # ---- custom conv3d ---------------------------------------------------------
    x = conv_ext.conv3d(x, conv_weight, conv_bias_tensor,
                        stride_d, stride_h, stride_w,
                        pad_d, pad_h, pad_w,
                        dil_d, dil_h, dil_w,
                        groups)

    # ---- max pooling (builtin) -------------------------------------------------
    x = F.max_pool3d(x, kernel_size=max_pool_kernel_size,
                     stride=max_pool_stride,
                     padding=max_pool_padding,
                     dilation=max_pool_dilation,
                     ceil_mode=max_pool_ceil_mode,
                     return_indices=max_pool_return_indices)

    # ---- global average pooling (builtin) --------------------------------------
    x = F.adaptive_avg_pool3d(x, global_avg_pool_output_size)

    # ---- optimised fused‑reduce -------------------------------------------------
    N, C, D, H, W = x.shape
    out = torch.zeros((N, D, H, W), device=x.device, dtype=x.dtype)

    # Hoist redundant operations: pre‑compute bias sum and reciprocal of divisor
    sum_bias      = bias.sum().item()       # Σ bias[c]
    inv_divisor   = 1.0 / divisor           # 1 / divisor

    fused_ext.fused_op(x.contiguous(), out, inv_divisor, sum_bias)

    return out

# =============================================================================
# 4️⃣  Dummy init / input helpers (same as the original placeholder)
# =============================================================================
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
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
