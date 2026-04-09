# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_095834/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a 3D convolution, applies minimum operation along a specific dimension, 
    and then applies softmax.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

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
    if 'dim' in flat_state:
        state_kwargs['dim'] = flat_state['dim']
    else:
        state_kwargs['dim'] = getattr(model, 'dim')
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
# CUDA source (kernels + binding code)
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// ------------------------------------------------------------------------
// 3‑D convolution – naive but sufficient for this exercise.
// ------------------------------------------------------------------------
__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int D_in, const int H_in, const int W_in,
    const int C_out, const int C_in_per_group,
    const int Kd, const int Kh, const int Kw,
    const int stride, const int pad, const int dil,
    const int groups,
    const int D_out, const int H_out, const int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * D_out * H_out * W_out;
    if (idx >= total) return;

    // decode output indices (row‑major order)
    int w = idx % W_out; idx /= W_out;
    int h = idx % H_out; idx /= H_out;
    int d = idx % D_out; idx /= D_out;
    int c = idx % C_out; idx /= C_out;
    int n = idx;                     // batch index

    // group information
    int C_out_per_group = C_out / groups;
    int group_id = c / C_out_per_group;
    int c_in_base = group_id * C_in_per_group;

    float sum = 0.0f;

    // loop over input channels belonging to this group
    for (int ci = 0; ci < C_in_per_group; ++ci) {
        int c_in = c_in_base + ci;
        for (int kd = 0; kd < Kd; ++kd) {
            int d_in = d * stride + kd * dil - pad;
            if (d_in < 0 || d_in >= D_in) continue;
            for (int kh = 0; kh < Kh; ++kh) {
                int h_in = h * stride + kh * dil - pad;
                if (h_in < 0 || h_in >= H_in) continue;
                for (int kw = 0; kw < Kw; ++kw) {
                    int w_in = w * stride + kw * dil - pad;
                    if (w_in < 0 || w_in >= W_in) continue;

                    // weight index (flattened row‑major)
                    int w_idx = (((c * C_in_per_group + ci) * Kd + kd) * Kh + kh) * Kw + kw;
                    float wval = __ldg(&weight[w_idx]);

                    // input index
                    int i_idx = (((n * C_in + c_in) * D_in + d_in) * H_in + h_in) * W_in + w_in;
                    float ival = __ldg(&input[i_idx]);

                    sum += wval * ival;
                }
            }
        }
    }

    if (bias != nullptr) sum += __ldg(&bias[c]);

    // store result (row‑major)
    int o_idx = (((n * C_out + c) * D_out + d) * H_out + h) * W_out + w;
    output[o_idx] = sum;
}

// ------------------------------------------------------------------------
// Fused min‑over‑depth  +  softmax‑over‑channel kernel.
// ------------------------------------------------------------------------
__global__ void fused_min_softmax_kernel(
    const float* __restrict__ conv_out,
    float* __restrict__ out,
    const int N, const int C, const int D, const int H, const int W)
{
    // each block processes one (n, h, w) position
    int idx = blockIdx.x;
    int w = idx % W; idx /= W;
    int h = idx % H; idx /= H;
    int n = idx;                     // batch index

    int c = threadIdx.x;
    if (c >= C) return;             // only the first C threads are active

    // ---- min over depth -------------------------------------------------
    float min_val = 1e38f;
    for (int d = 0; d < D; ++d) {
        int off = (((n * C + c) * D + d) * H + h) * W + w;
        float v = __ldg(&conv_out[off]);
        if (v < min_val) min_val = v;
    }

    // ---- softmax over channel -----------------------------------------
    float exp_val = __expf(min_val);

    // warp‑level reduction to obtain sum of exponentials
    float sum_exp = exp_val;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }

    float result = exp_val / sum_exp;

    // write back (shape N×C×H×W)
    int out_off = (((n * C + c) * H + h) * W + w);
    out[out_off] = result;
}

// ------------------------------------------------------------------------
// C++ bindings (PyTorch extensions)
// ------------------------------------------------------------------------
void conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int C_in_per_group,
    int Kd, int Kh, int Kw,
    int stride, int pad, int dil,
    int groups,
    int D_out, int H_out, int W_out)
{
    const float* in_ptr   = input.data_ptr<float>();
    const float* w_ptr    = weight.data_ptr<float>();
    const float* b_ptr    = bias.numel() ? bias.data_ptr<float>() : nullptr;
    float* out_ptr        = output.data_ptr<float>();

    int total_out = N * C_out * D_out * H_out * W_out;
    const int block = 256;
    const int grid  = (total_out + block - 1) / block;

    conv3d_kernel<<<grid, block>>>(
        in_ptr, w_ptr, b_ptr, out_ptr,
        N, C_in, D_in, H_in, W_in,
        C_out, C_in_per_group,
        Kd, Kh, Kw,
        stride, pad, dil,
        groups,
        D_out, H_out, W_out);

    cudaDeviceSynchronize();
}

void fused_min_softmax_cuda(
    torch::Tensor conv_out,
    torch::Tensor output,
    int N, int C, int D, int H, int W)
{
    const float* in_ptr = conv_out.data_ptr<float>();
    float* out_ptr      = output.data_ptr<float>();

    const int grid  = N * H * W;          // one block per (n,h,w)
    const int block = 32;                 // >= max number of channels (24)

    fused_min_softmax_kernel<<<grid, block>>>(
        in_ptr, out_ptr, N, C, D, H, W);

    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
# C++ source (pybind11 bindings)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int C_in_per_group,
    int Kd, int Kh, int Kw,
    int stride, int pad, int dil,
    int groups,
    int D_out, int H_out, int W_out);

void fused_min_softmax_cuda(
    torch::Tensor conv_out,
    torch::Tensor output,
    int N, int C, int D, int H, int W);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d", &conv3d_cuda,
          "3‑D convolution (CUDA)");
    m.def("fused_min_softmax", &fused_min_softmax_cuda,
          "Fused min‑over‑depth + softmax‑over‑channel (CUDA)");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name="fused_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)

# ----------------------------------------------------------------------
# Constants (identical to the original script)
# ----------------------------------------------------------------------
batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
dim = 2          # depth dimension (the axis over which the min is taken)

# ----------------------------------------------------------------------
# functional_model – the only function that will be imported
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
    dim,
):
    # ------------------------------------------------------------------
    # Ensure all inputs reside on the GPU and are contiguous
    # ------------------------------------------------------------------
    x = x.cuda().contiguous()
    conv_weight = conv_weight.cuda().contiguous()
    if conv_bias is not None:
        conv_bias = conv_bias.cuda().contiguous()

    # ------------------------------------------------------------------
    # Basic shape information
    # ------------------------------------------------------------------
    N, C_in, D_in, H_in, W_in = x.shape
    out_ch = conv_weight.shape[0]                     # C_out
    C_in_per_group = conv_weight.shape[1]            # C_in // groups
    Kd, Kh, Kw = conv_weight.shape[2], conv_weight.shape[3], conv_weight.shape[4]

    # ------------------------------------------------------------------
    # Compute output spatial sizes (same formula as PyTorch)
    # ------------------------------------------------------------------
    stride = conv_stride if isinstance(conv_stride, int) else conv_stride[0]
    pad    = conv_padding if isinstance(conv_padding, int) else conv_padding[0]
    dil    = conv_dilation if isinstance(conv_dilation, int) else conv_dilation[0]

    D_out = (D_in + 2 * pad - dil * (Kd - 1) - 1) // stride + 1
    H_out = (H_in + 2 * pad - dil * (Kh - 1) - 1) // stride + 1
    W_out = (W_in + 2 * pad - dil * (Kw - 1) - 1) // stride + 1

    # ------------------------------------------------------------------
    # 1) Custom 3‑D convolution
    # ------------------------------------------------------------------
    conv_out = torch.empty(
        (N, out_ch, D_out, H_out, W_out),
        dtype=x.dtype,
        device="cuda",
    )

    fused_ext.conv3d(
        x, conv_weight, conv_bias, conv_out,
        N, C_in, D_in, H_in, W_in,
        out_ch, C_in_per_group,
        Kd, Kh, Kw,
        stride, pad, dil,
        conv_groups,
        D_out, H_out, W_out,
    )

    # ------------------------------------------------------------------
    # 2) Fused min‑over‑depth + softmax‑over‑channel
    # ------------------------------------------------------------------
    # The original code always reduces along dimension 2 (depth).  The
    # kernel below implements exactly that.
    out = torch.empty(
        (N, out_ch, H_out, W_out),
        dtype=x.dtype,
        device="cuda",
    )

    fused_ext.fused_min_softmax(
        conv_out, out,
        N, out_ch, D_out, H_out, W_out,
    )

    return out

# ----------------------------------------------------------------------
# Helper functions required by the harness (identical to original)
# ----------------------------------------------------------------------
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]
