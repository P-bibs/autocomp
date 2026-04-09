# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123254/code_15.py
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
# Inline CUDA source – fused transposed‑conv + bias + tanh kernel
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_transpose_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,   // bias added inside the conv
    const float* __restrict__ bias,        // bias subtracted after the conv
    const int N, const int C_in, const int C_out,
    const int H, const int W,
    const int K,                           // kernel size (square)
    const int stride_h, const int stride_w,
    const int pad_h,   const int pad_w,
    const int dilation_h, const int dilation_w,
    const int groups,
    const int out_h, const int out_w,
    float* __restrict__ output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * out_h * out_w) return;

    // decode flat index -> (n, co, y, x)
    int n = idx / (C_out * out_h * out_w);
    int rem = idx % (C_out * out_h * out_w);
    int co = rem / (out_h * out_w);
    int rem2 = rem % (out_h * out_w);
    int y = rem2 / out_w;
    int x = rem2 % out_w;

    // ----- per‑output channel group information -----
    const int C_in_per_group  = C_in  / groups;
    const int C_out_per_group = C_out / groups;
    const int group_id   = co / C_out_per_group;
    const int ci_start   = group_id * C_in_per_group;
    const int ci_end     = ci_start + C_in_per_group;

    float sum = 0.0f;

    // ----- gather‑style transposed convolution -----
    for (int ci = ci_start; ci < ci_end; ++ci) {
        // weight offset for (co_local, ci_local)
        int weight_ci_offset = ((co % C_out_per_group) * C_in_per_group + (ci - ci_start)) * (K * K);
        for (int kh = 0; kh < K; ++kh) {
            int y_i = y + pad_h - kh * dilation_h;
            if (y_i < 0 || y_i % stride_h != 0) continue;
            int i = y_i / stride_h;
            if (i >= H) continue;
            for (int kw = 0; kw < K; ++kw) {
                int x_j = x + pad_w - kw * dilation_w;
                if (x_j < 0 || x_j % stride_w != 0) continue;
                int j = x_j / stride_w;
                if (j >= W) continue;

                // input[n, ci, i, j]
                int in_idx = ((n * C_in + ci) * H + i) * W + j;
                // weight[co_local, ci_local, kh, kw]
                int w_idx = weight_ci_offset + kh * K + kw;

                sum += input[in_idx] * weight[w_idx];
            }
        }
    }

    // ----- bias (conv bias) + extra bias + tanh -----
    sum += conv_bias[co];   // bias passed to the original conv_transpose2d
    sum -= bias[co];        // the bias that is subtracted afterwards
    float out_val = tanhf(sum);

    // store result
    int out_idx = ((n * C_out + co) * out_h + y) * out_w + x;
    output[out_idx] = out_val;
}

/* Host‑side wrapper that launches the kernel */
void fused_op_forward(
    at::Tensor input,    // (N, C_in, H, W)
    at::Tensor weight,   // (C_out, C_in/groups, K, K)
    at::Tensor conv_bias,
    at::Tensor bias,
    int N, int C_in, int C_out,
    int H, int W,
    int K,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups,
    int out_h, int out_w,
    at::Tensor output)   // (N, C_out, out_h, out_w)
{
    const int threads = 256;
    const int total   = N * C_out * out_h * out_w;
    const int blocks  = (total + threads - 1) / threads;

    fused_transpose_conv_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C_in, C_out, H, W,
        K,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        groups,
        out_h, out_w,
        output.data_ptr<float>());

    cudaDeviceSynchronize();   // ensure kernel completion before returning
}
"""

# -------------------------------------------------------------------------
# C++ binding (PyBind11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor conv_bias,
    at::Tensor bias,
    int N, int C_in, int C_out,
    int H, int W,
    int K,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups,
    int out_h, int out_w,
    at::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward,
          "Fused transposed convolution + bias subtraction + tanh");
}
"""

# -------------------------------------------------------------------------
# Compile the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Helper to turn a stride/padding/etc. that may be int or (a,b) into (a,b)
# -------------------------------------------------------------------------
def _parse_pair(val, name):
    if isinstance(val, (int, float)):
        return int(val), int(val)
    elif isinstance(val, (tuple, list)):
        if len(val) == 2:
            return int(val[0]), int(val[1])
        else:
            return int(val[0]), int(val[0])
    else:
        raise ValueError(f"Unexpected type for {name}: {type(val)}")

# -------------------------------------------------------------------------
# The optimized functional_model
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
    # -----------------------------------------------------------------
    # Ensure all tensors are on the GPU
    # -----------------------------------------------------------------
    if not x.is_cuda:
        x = x.cuda()
    if not conv_transpose_weight.is_cuda:
        conv_transpose_weight = conv_transpose_weight.cuda()
    if conv_transpose_bias is not None and not conv_transpose_bias.is_cuda:
        conv_transpose_bias = conv_transpose_bias.cuda()
    if not bias.is_cuda:
        bias = bias.cuda()

    # -----------------------------------------------------------------
    # Parse convolution parameters (they can be int or 2‑tuple)
    # -----------------------------------------------------------------
    stride_h, stride_w = _parse_pair(conv_transpose_stride, "stride")
    pad_h,   pad_w    = _parse_pair(conv_transpose_padding, "padding")
    out_pad_h, out_pad_w = _parse_pair(conv_transpose_output_padding, "output_padding")
    dilation_h, dilation_w = _parse_pair(conv_transpose_dilation, "dilation")
    groups = int(conv_transpose_groups)

    # -----------------------------------------------------------------
    # Input / weight shapes
    # -----------------------------------------------------------------
    N, C_in, H, W = x.shape                       # (N, C_in, H, W)
    C_out = conv_transpose_weight.shape[0]        # (C_out, C_in/groups, K, K)
    K = conv_transpose_weight.shape[2]            # assume square kernel

    # -----------------------------------------------------------------
    # Compute output spatial size using the standard deconv formula
    # -----------------------------------------------------------------
    out_h = (H - 1) * stride_h - 2 * pad_h + dilation_h * (K - 1) + out_pad_h + 1
    out_w = (W - 1) * stride_w - 2 * pad_w + dilation_w * (K - 1) + out_pad_w + 1

    # -----------------------------------------------------------------
    # If the user did not provide a conv‑bias, substitute a zero tensor
    # -----------------------------------------------------------------
    if conv_transpose_bias is None:
        conv_bias = torch.zeros(C_out, dtype=x.dtype, device=x.device)
    else:
        conv_bias = conv_transpose_bias

    # -----------------------------------------------------------------
    # Allocate output and launch the fused kernel
    # -----------------------------------------------------------------
    out = torch.empty((N, C_out, out_h, out_w), dtype=x.dtype, device=x.device)

    fused_ext.fused_op(
        x,                     # input
        conv_transpose_weight, # weight
        conv_bias,             # bias added inside the conv
        bias,                  # bias subtracted afterwards
        N, C_in, C_out,
        H, W,
        K,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        groups,
        out_h, out_w,
        out                    # result
    )
    return out
