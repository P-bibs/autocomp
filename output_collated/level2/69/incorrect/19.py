# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_051539/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies HardSwish, and then ReLU.
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
# CUDA source – fused conv + hardswish + relu kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_act_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N,                 // total number of output elements (batch * out_ch * out_h * out_w)
    const int batch,
    const int in_ch,
    const int out_ch,
    const int in_h,
    const int in_w,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int out_h,
    const int out_w)
{
    // Grid‑stride loop – each thread handles multiple output positions
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
        // Decode flat index to (batch, out_c, out_y, out_x)
        int tmp = idx;
        int out_c = tmp % out_ch;
        tmp /= out_ch;
        int out_y = tmp % out_h;
        int out_x = tmp % out_w;
        tmp /= out_w;
        int b = tmp;               // batch index

        // ---------- naive convolution ----------
        float sum = 0.0f;

        // grouping support
        int in_ch_per_group  = in_ch  / groups;
        int out_ch_per_group = out_ch / groups;
        int group_id = out_c / out_ch_per_group;
        int in_ch_start = group_id * in_ch_per_group;

        // loop over input channels belonging to this group
        for (int ic = 0; ic < in_ch_per_group; ++ic) {
            int ic_global = in_ch_start + ic;
            // kernel height
            for (int ky = 0; ky < kernel_size; ++ky) {
                int iy = out_y * stride - padding + ky * dilation;
                if (iy < 0 || iy >= in_h) continue;
                // kernel width
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int ix = out_x * stride - padding + kx * dilation;
                    if (ix < 0 || ix >= in_w) continue;

                    // weight index: (out_c, ic, ky, kx) stored as (out_c, ic, ky*kernel_size + kx)
                    int weight_idx = ((out_c * in_ch_per_group) + ic) * (kernel_size * kernel_size)
                                     + ky * kernel_size + kx;
                    float w = weight[weight_idx];

                    // input index: (b, ic_global, iy, ix)
                    int input_idx = ((b * in_ch + ic_global) * in_h + iy) * in_w + ix;
                    float inp = input[input_idx];

                    sum += inp * w;
                }
            }
        }

        // add bias if present
        if (bias != nullptr) {
            sum += bias[out_c];
        }

        // ---------- activations ----------
        // hardswish: x * ReLU6(x+3) / 6
        float hs = sum * fminf(fmaxf(sum + 3.0f, 0.0f), 6.0f) / 6.0f;
        // ReLU
        float out_val = fmaxf(0.0f, hs);

        // write final result
        int out_idx = ((b * out_ch + out_c) * out_h + out_y) * out_w + out_x;
        output[out_idx] = out_val;
    }
}

// Launcher wrapper – called from Python
void fused_conv_act(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch,
    int in_ch,
    int out_ch,
    int in_h,
    int in_w,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    int out_h,
    int out_w,
    int threads,
    int blocks,
    torch::Tensor output)
{
    int N = batch * out_ch * out_h * out_w;
    const float* input_ptr  = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr   = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;
    float*       output_ptr = output.data_ptr<float>();

    fused_conv_act_kernel<<<blocks, threads>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        N, batch, in_ch, out_ch, in_h, in_w,
        kernel_size, stride, padding, dilation, groups,
        out_h, out_w);
}
"""

# ----------------------------------------------------------------------
# C++ binding – exposes the launcher to Python
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_act(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch,
    int in_ch,
    int out_ch,
    int in_h,
    int in_w,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    int out_h,
    int out_w,
    int threads,
    int blocks,
    torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_act", &fused_conv_act,
          "Fused convolution + hardswish + ReLU kernel");
}
"""

# ----------------------------------------------------------------------
# Compile the CUDA extension (runs once at import time)
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_conv_act',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# ----------------------------------------------------------------------
# The functional_model that will be imported / evaluated
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
    # ------------------------------------------------------------------
    # Move tensors to GPU if they are not already there
    # ------------------------------------------------------------------
    if not x.is_cuda:
        x = x.cuda()
    if not conv_weight.is_cuda:
        conv_weight = conv_weight.cuda()

    # If no bias is supplied we create an empty (size‑0) tensor; the kernel
    # will treat a zero‑size bias as “no bias”.
    if conv_bias is not None:
        if not conv_bias.is_cuda:
            conv_bias = conv_bias.cuda()
    else:
        conv_bias = torch.empty(0, dtype=x.dtype, device='cuda')

    # ------------------------------------------------------------------
    # Extract / compute all required dimensions
    # ------------------------------------------------------------------
    batch, in_ch, in_h, in_w = x.shape                     # (N, C, H, W)
    out_ch = conv_weight.shape[0]                          # (F, C, kH, kW)
    kernel_size = conv_weight.shape[2]                     # square kernel assumed
    stride = conv_stride
    padding = conv_padding
    dilation = conv_dilation
    groups = conv_groups

    # Output spatial size (forward convolution formula)
    out_h = (in_h + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    # Allocate output tensor on the GPU
    output = torch.empty((batch, out_ch, out_h, out_w), dtype=x.dtype, device='cuda')

    # ------------------------------------------------------------------
    # Launch configuration – fixed thread‑block size, enough blocks to
    # cover the whole output, but cap to a reasonable maximum (4096).
    # ------------------------------------------------------------------
    N = batch * out_ch * out_h * out_w
    threads_per_block = 256
    blocks = (N + threads_per_block - 1) // threads_per_block
    max_blocks = 4096
    if blocks > max_blocks:
        blocks = max_blocks

    # ------------------------------------------------------------------
    # Call the fused CUDA kernel
    # ------------------------------------------------------------------
    fused_ext.fused_conv_act(
        x,                      # input
        conv_weight,            # conv weights
        conv_bias,              # bias (or empty)
        batch,
        in_ch,
        out_ch,
        in_h,
        in_w,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        out_h,
        out_w,
        threads_per_block,
        blocks,
        output,                 # output tensor (filled in‑place)
    )

    return output
