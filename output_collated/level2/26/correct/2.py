# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_035600/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x', 'add_input']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
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
    # State for conv_transpose (nn.ConvTranspose3d)
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
# CUDA source – fused transposed‑conv + add + HardSwish
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float hardswish(float x) {
    float y = x + 3.0f;
    y = fminf(fmaxf(y, 0.0f), 6.0f);
    return x * y / 6.0f;
}

__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ input,      // (N, C_in, D_in, H_in, W_in)
    const float* __restrict__ weight,     // (C_in, C_out/group, K, K, K)
    const float* __restrict__ add_input,  // (N, C_out, D_out, H_out, W_out)
    const float* __restrict__ bias,       // (C_out,)  (may be nullptr)
    float* __restrict__ output,           // (N, C_out, D_out, H_out, W_out)
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int stride, const int padding, const int kernel_size,
    const int output_padding, const int dilation, const int groups)
{
    const int K = kernel_size;
    const int K3 = K * K * K;
    const int group_size = C_in / groups;
    const int out_group_size = C_out / groups;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_out = N * C_out * D_out * H_out * W_out;
    if (idx >= total_out) return;

    // ---- decode linear index to (n, oc, od, oh, ow) ----
    int rem = idx;
    const int c_out_dim = C_out * D_out * H_out * W_out;
    int n = rem / c_out_dim;
    rem = rem % c_out_dim;
    int oc = rem / (D_out * H_out * W_out);
    rem = rem % (D_out * H_out * W_out);
    int od = rem / (H_out * W_out);
    rem = rem % (H_out * W_out);
    int oh = rem / W_out;
    int ow = rem % W_out;

    // ---- conv_transpose (gather formulation) ----
    float conv = 0.0f;

    int group_id = oc / out_group_size;
    int oc_in_group = oc % out_group_size;
    
    for (int ic_in_group = 0; ic_in_group < group_size; ++ic_in_group) {
        int ic = group_id * group_size + ic_in_group;
        
        // base offsets
        const int w_base = ((ic_in_group * out_group_size + oc_in_group) * K3);
        const int in_base = ((n * C_in + ic) * D_in * H_in * W_in);

        #pragma unroll
        for (int kz = 0; kz < K; ++kz) {
            int i_d = (od + padding - kz * dilation);
            if (i_d % stride != 0) continue;
            i_d /= stride;
            if (i_d < 0 || i_d >= D_in) continue;

            const int off_kz = kz * K * K;
            #pragma unroll
            for (int ky = 0; ky < K; ++ky) {
                int i_h = (oh + padding - ky * dilation);
                if (i_h % stride != 0) continue;
                i_h /= stride;
                if (i_h < 0 || i_h >= H_in) continue;

                const int off_ky = off_kz + ky * K;
                #pragma unroll
                for (int kx = 0; kx < K; ++kx) {
                    int i_w = (ow + padding - kx * dilation);
                    if (i_w % stride != 0) continue;
                    i_w /= stride;
                    if (i_w < 0 || i_w >= W_in) continue;

                    int w_idx = w_base + off_ky + kx;
                    float w = weight[w_idx];

                    int i_idx = in_base + ((i_d * H_in + i_h) * W_in + i_w);
                    conv += w * input[i_idx];
                }
            }
        }
    }

    if (bias) conv += bias[oc];

    // ---- add_input and HardSwish ----
    int out_idx = ((n * C_out + oc) * D_out + od) * H_out * W_out + oh * W_out + ow;
    float a = add_input[out_idx];
    float val = conv + a;
    float act = hardswish(val);
    output[out_idx] = val * act;
}

// --------------------------------------------------------------------
// Host wrapper that launches the kernel
// --------------------------------------------------------------------
void fused_op(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor add_input,
    at::Tensor bias,
    at::Tensor output,
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int stride, int padding, int kernel_size,
    int output_padding, int dilation, int groups)
{
    const int threads = 256;
    const int blocks = (N * C_out * D_out * H_out * W_out + threads - 1) / threads;

    fused_conv_transpose_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        add_input.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride, padding, kernel_size,
        output_padding, dilation, groups);
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor add_input,
    at::Tensor bias,
    at::Tensor output,
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int stride, int padding, int kernel_size,
    int output_padding, int dilation, int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op,
          "Fused transposed convolution + add + HardSwish");
}
"""

# -------------------------------------------------------------------------
# Compile the fused CUDA extension (will be loaded on first import)
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)


# -------------------------------------------------------------------------
# The functional model that will be evaluated
# -------------------------------------------------------------------------
def functional_model(
    x,
    add_input,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    bias,  # not used in the original program
):
    # ---- unpack geometry -------------------------------------------------
    stride = conv_transpose_stride if isinstance(conv_transpose_stride, int) else conv_transpose_stride[0]
    padding = conv_transpose_padding if isinstance(conv_transpose_padding, int) else conv_transpose_padding[0]
    output_padding = conv_transpose_output_padding if isinstance(conv_transpose_output_padding, int) else conv_transpose_output_padding[0]
    dilation = conv_transpose_dilation if isinstance(conv_transpose_dilation, int) else conv_transpose_dilation[0]
    groups = conv_transpose_groups if isinstance(conv_transpose_groups, int) else conv_transpose_groups[0]
    kernel_size = conv_transpose_weight.shape[2]           # K

    N, C_in, D_in, H_in, W_in = x.shape
    C_out = conv_transpose_weight.shape[1]                # out channels

    # ---- output spatial size (standard conv_transpose formula) ----------
    D_out = (D_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

    # ---- allocate result tensor -----------------------------------------
    output = torch.empty((N, C_out, D_out, H_out, W_out),
                         dtype=x.dtype, device=x.device)

    # ---- launch fused kernel --------------------------------------------
    fused_ext.fused_op(
        x,
        conv_transpose_weight,
        add_input,
        conv_transpose_bias,          # may be None → handled inside kernel
        output,
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride, padding, kernel_size,
        output_padding, dilation, groups,
    )
    return output


# -------------------------------------------------------------------------
# Helper functions required by the harness (kept for completeness)
# -------------------------------------------------------------------------
def get_init_inputs():
    # Original signature, not used by the fused kernel but needed by the test harness
    return [
        32,   # in_channels
        64,   # out_channels
        3,    # kernel_size
        2,    # stride
        1,    # padding
        1,    # output_padding
        (64, 1, 1, 1, 1),  # bias shape (not actually used)
    ]


def get_inputs():
    batch_size = 128
    in_channels = 32
    out_channels = 64
    D = H = W = 16
    # input tensor
    x = torch.rand(batch_size, in_channels, D, H, W)
    # tensor to be added after convolution
    add_input = torch.rand(batch_size, out_channels, D * 2, H * 2, W * 2)
    return [x, add_input]
