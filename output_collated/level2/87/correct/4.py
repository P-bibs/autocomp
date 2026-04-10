# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141921/code_6.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'subtract_value_1', 'subtract_value_2']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'subtract_value_1', 'subtract_value_2']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, subtracts two values, applies Mish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

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
    if 'subtract_value_1' in flat_state:
        state_kwargs['subtract_value_1'] = flat_state['subtract_value_1']
    else:
        state_kwargs['subtract_value_1'] = getattr(model, 'subtract_value_1')
    if 'subtract_value_2' in flat_state:
        state_kwargs['subtract_value_2'] = flat_state['subtract_value_2']
    else:
        state_kwargs['subtract_value_2'] = getattr(model, 'subtract_value_2')
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
# CUDA source – tile-based convolution with shared-memory weight cache
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch, const int in_c, const int in_h, const int in_w,
    const int out_c, const int k,
    const int out_h, const int out_w,
    const float sub1, const float sub2,
    const int tile_h, const int tile_w,
    const int tile_y_cnt, const int tile_x_cnt) {

    /* ------------------------------------------------------------------
       Decompose block index:
       blockIdx.x = b * out_c * tile_y_cnt * tile_x_cnt
                    + oc * tile_y_cnt * tile_x_cnt
                    + tile_y * tile_x_cnt
                    + tile_x
       ------------------------------------------------------------------ */
    int idx = blockIdx.x;
    int tile_x = idx % tile_x_cnt; idx /= tile_x_cnt;
    int tile_y = idx % tile_y_cnt; idx /= tile_y_cnt;
    int oc    = idx % out_c;       idx /= out_c;
    int b     = idx;                // batch index

    // spatial origin of the 8×8 tile processed by this block
    int oh_base = tile_y * tile_h;
    int ow_base = tile_x * tile_w;

    // thread's position inside the tile
    int local_id = threadIdx.x;
    if (local_id >= tile_h * tile_w) return;   // safety

    int oh = oh_base + local_id / tile_w;
    int ow = ow_base + local_id % tile_w;

    // out-of-bound guard (needed for last incomplete tile)
    if (oh >= out_h || ow >= out_w) return;

    // ------------------------------------------------------------------
    // Shared memory for the weight of the current output channel
    // ------------------------------------------------------------------
    extern __shared__ float s_weight[];   // size = in_c * k * k

    // load weight (only thread 0 does the copy)
    if (threadIdx.x == 0) {
        int w_off = oc * in_c * k * k;
        for (int i = 0; i < in_c * k * k; ++i)
            s_weight[i] = weight[w_off + i];
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Convolution (same arithmetic as the original kernel)
    // ------------------------------------------------------------------
    float acc = bias[oc];
    for (int ic = 0; ic < in_c; ++ic) {
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                float in_val = input[((b * in_c + ic) * in_h + (oh + i)) * in_w + (ow + j)];
                float w_val  = s_weight[((ic * k + i) * k + j)];
                acc += in_val * w_val;
            }
        }
    }

    // subtraction & Mish activation
    float val = acc - sub1 - sub2;
    output[((b * out_c + oc) * out_h + oh) * out_w + ow] =
        val * tanhf(logf(1.0f + expf(val)));
}

/* ----------------------------------------------------------------------
   Host wrapper – computes launch parameters and passes them to the kernel
   ---------------------------------------------------------------------- */
void fused_conv_mish(torch::Tensor input, torch::Tensor weight,
                     torch::Tensor bias, torch::Tensor output,
                     float sub1, float sub2) {
    const int batch   = input.size(0);
    const int in_c    = input.size(1);
    const int in_h    = input.size(2);
    const int in_w    = input.size(3);
    const int out_c   = weight.size(0);
    const int k       = weight.size(2);
    const int out_h   = in_h - k + 1;
    const int out_w   = in_w - k + 1;

    const int tile_h = 8;
    const int tile_w = 8;
    const int tile_y_cnt = (out_h + tile_h - 1) / tile_h;
    const int tile_x_cnt = (out_w + tile_w - 1) / tile_w;

    const int blocks  = batch * out_c * tile_y_cnt * tile_x_cnt;
    const int threads = tile_h * tile_w;                 // 64
    const int shared_mem = in_c * k * k * sizeof(float); // weight cache

    fused_conv_mish_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.data_ptr<float>(), output.data_ptr<float>(),
        batch, in_c, in_h, in_w, out_c, k,
        out_h, out_w, sub1, sub2,
        tile_h, tile_w, tile_y_cnt, tile_x_cnt);
}
"""

# ----------------------------------------------------------------------
# C++ binding (pybind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_conv_mish(torch::Tensor i, torch::Tensor w,
                     torch::Tensor b, torch::Tensor o,
                     float s1, float s2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_mish", &fused_conv_mish,
          "Fused Conv + Subtract + Mish (CUDA)");
}
"""

# ----------------------------------------------------------------------
# Build the inline extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional wrapper – matches the expected signature
# ----------------------------------------------------------------------
def functional_model(x, *, conv_weight, conv_bias,
                     conv_stride=1, conv_padding=0,
                     conv_dilation=1, conv_groups=1,
                     subtract_value_1, subtract_value_2):
    """
    Expects stride=1, padding=0, dilation=1, groups=1 (square kernel).
    Returns the fused Conv → subtract → Mish result.
    """
    batch, _, h, w = x.shape
    k = conv_weight.shape[2]                # square kernel side
    out_h = h - k + 1
    out_w = w - k + 1

    out = torch.empty((batch, conv_weight.size(0), out_h, out_w),
                      device=x.device, dtype=x.dtype)

    fused_ext.fused_conv_mish(x, conv_weight, conv_bias, out,
                              subtract_value_1, subtract_value_2)
    return out
