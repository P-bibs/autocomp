# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_144040/code_13.py
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




import math
import torch
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# Optimized CUDA kernel: weight + input tiling (shared memory)
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
    const int tile_h, const int tile_w) {

    // Shared memory layout:
    //   s_weight[ in_c * k * k ]               (weight cache)
    //   s_input[ in_c * region_h * region_w ]  (input tile)
    extern __shared__ float s_mem[];

    const int weight_size = in_c * k * k;
    float* s_weight = s_mem;
    float* s_input  = s_mem + weight_size;

    // ---- 1. load weight for this output channel --------------------
    const int oc = blockIdx.y;          // output channel
    const int b  = blockIdx.z;          // batch element

    for (int idx = threadIdx.x; idx < weight_size; idx += blockDim.x) {
        s_weight[idx] = weight[oc * weight_size + idx];
    }
    __syncthreads();

    // ---- 2. Determine tile location ---------------------------------
    const int tiles_x = (out_w + tile_w - 1) / tile_w;
    const int tile_id = blockIdx.x;
    const int tile_y  = tile_id / tiles_x;
    const int tile_x  = tile_id % tiles_x;

    const int out_start_h = tile_y * tile_h;
    const int out_start_w = tile_x * tile_w;

    const int region_h = tile_h + k - 1;
    const int region_w = tile_w + k - 1;
    const int region_size = in_c * region_h * region_w;

    // ---- 3. load input tile into shared memory ----------------------
    for (int idx = threadIdx.x; idx < region_size; idx += blockDim.x) {
        int rem = idx;
        const int ic = rem / (region_h * region_w);
        rem = rem % (region_h * region_w);
        const int ih = rem / region_w;
        const int iw = rem % region_w;

        int h = out_start_h + ih;
        int w = out_start_w + iw;
        // clamp to valid input range
        if (h >= in_h) h = in_h - 1;
        if (w >= in_w) w = in_w - 1;

        const int input_idx = ((b * in_c + ic) * in_h + h) * in_w + w;
        s_input[(ic * region_h + ih) * region_w + iw] = input[input_idx];
    }
    __syncthreads();

    // ---- 4. compute output for the thread's pixel -------------------
    const int tile_sz = tile_h * tile_w;
    const int local_id = threadIdx.x;
    if (local_id >= tile_sz) return;

    const int oh = out_start_h + local_id / tile_w;
    const int ow = out_start_w + local_id % tile_w;
    if (oh >= out_h || ow >= out_w) return;

    float acc = bias[oc];

    const int i_offset = oh - out_start_h;
    const int j_offset = ow - out_start_w;

    for (int ic = 0; ic < in_c; ++ic) {
        const float* const s_in_ic = s_input + ic * region_h * region_w;
        for (int pi = 0; pi < k; ++pi) {
            for (int pj = 0; pj < k; ++pj) {
                const int ii = i_offset + pi;
                const int jj = j_offset + pj;
                const float in_val = s_in_ic[ii * region_w + jj];
                const float w_val  = s_weight[((ic * k + pi) * k + pj)];
                acc += in_val * w_val;
            }
        }
    }

    // ---- 5. fused Mish activation ------------------------------------
    const float val = acc - sub1 - sub2;
    const float out_val = val * tanhf(logf(1.0f + expf(val)));

    output[((b * out_c + oc) * out_h + oh) * out_w + ow] = out_val;
}

void fused_conv_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                     torch::Tensor output, float sub1, float sub2,
                     int tile_h, int tile_w) {
    const int batch = input.size(0);
    const int in_c  = input.size(1);
    const int in_h  = input.size(2);
    const int in_w  = input.size(3);
    const int out_c = weight.size(0);
    const int k     = weight.size(2);
    const int out_h = in_h - k + 1;
    const int out_w = in_w - k + 1;

    // Determine number of tiles
    const int tiles_x = (out_w + tile_w - 1) / tile_w;
    const int tiles_y = (out_h + tile_h - 1) / tile_h;
    const int total_tiles = tiles_x * tiles_y;

    const dim3 grid(total_tiles, out_c, batch);
    const int threads = 256;                 // enough parallelism for weight & input loads

    // shared memory: weight + input tile
    const int weight_size = in_c * k * k;
    const int region_h = tile_h + k - 1;
    const int region_w = tile_w + k - 1;
    const int region_size = in_c * region_h * region_w;
    const size_t shared_bytes = (weight_size + region_size) * sizeof(float);

    fused_conv_mish_kernel<<<grid, threads, shared_bytes>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch, in_c, in_h, in_w,
        out_c, k, out_h, out_w, sub1, sub2, tile_h, tile_w);
}
"""

# ----------------------------------------------------------------------
# C++ binding (pybind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_conv_mish(torch::Tensor i, torch::Tensor w, torch::Tensor b,
                     torch::Tensor o, float s1, float s2, int tile_h, int tile_w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_mish", &fused_conv_mish,
          "Fused convolution + Mish (weight + input tiling)");
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
# Functional wrapper
# ----------------------------------------------------------------------
def functional_model(x, *, conv_weight, conv_bias,
                     conv_stride=1, conv_padding=0,
                     conv_dilation=1, conv_groups=1,
                     subtract_value_1, subtract_value_2):
    """
    Expected use: stride=1, padding=0, dilation=1, groups=1.
    The kernel computes out_h = in_h - k + 1, out_w = in_w - k + 1.
    """
    batch, in_c, in_h, in_w = x.shape
    out_c = conv_weight.shape[0]
    k = conv_weight.shape[2]          # square kernel assumed
    out_h = in_h - k + 1
    out_w = in_w - k + 1

    # ------------------------------------------------------------------
    # Choose a tile size that fits into 48 KB of shared memory.
    # weight_size (floats) + input_region_size (floats) ≤ 48 KB / 4
    # ------------------------------------------------------------------
    weight_size = in_c * k * k
    max_shared_bytes = 48 * 1024                      # 48 KB
    max_input_bytes = max_shared_bytes - weight_size * 4   # 4 bytes/float
    max_input_elems = max_input_bytes // 4                # floats
    region_elem_max = max_input_elems // in_c

    # tile side = floor(sqrt(region_elem_max)) - (k-1), at least 1
    tile_side = int(math.sqrt(region_elem_max)) - (k - 1)
    if tile_side < 1:
        tile_side = 1
    # clamp to output dimensions
    if tile_side > out_h:
        tile_side = out_h
    if tile_side > out_w:
        tile_side = out_w

    tile_h = tile_side
    tile_w = tile_side

    # allocate output tensor
    out = torch.empty((batch, out_c, out_h, out_w),
                      dtype=torch.float32, device=x.device)

    fused_ext.fused_conv_mish(x, conv_weight, conv_bias, out,
                              subtract_value_1, subtract_value_2,
                              tile_h, tile_w)
    return out
