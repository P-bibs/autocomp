# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150347/code_11.py
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




# -*- coding: utf-8 -*-
import torch
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------------------
# CUDA kernel: shared‑memory caching for *both* weights and input tiles
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE 16                 // output tile per block (X × Y)
#define BANK_PAD 1              // 1‑float padding to avoid bank conflicts

// ---------------------------------------------------------------------
//  Kernel
// ---------------------------------------------------------------------
extern "C" __global__
void fused_conv_mish_kernel(
    const float* __restrict__ input,   // NCHW
    const float* __restrict__ weight,  // OC × K × K × IC   (packed as OC, K*K*IC)
    const float* __restrict__ bias,    // OC
    float* __restrict__ output,        // NCHW (output)
    // tensor sizes
    const int batch,
    const int in_c,
    const int in_h,
    const int in_w,
    const int out_c,
    const int k,
    const int out_h,
    const int out_w,
    // Mish constants
    const float sub1,
    const float sub2)
{
    // -----------------------------------------------------------------
    //  Block / thread indices
    // -----------------------------------------------------------------
    const int oc = blockIdx.y;                 // output channel
    const int b  = blockIdx.z;                 // batch index

    // blockIdx.x encodes the (tile_row, tile_col) pair
    const int tiles_x = (out_w + TILE - 1) / TILE;
    const int tile_row = blockIdx.x / tiles_x;
    const int tile_col = blockIdx.x % tiles_x;

    const int ty = threadIdx.y;                // thread row inside the tile
    const int tx = threadIdx.x;                // thread col inside the tile

    const int out_y = tile_row * TILE + ty;    // output pixel y (height)
    const int out_x = tile_col * TILE + tx;    // output pixel x (width)

    // -----------------------------------------------------------------
    //  Shared memory layout
    // -----------------------------------------------------------------
    // 0 … weight_vol-1                : weight buffer
    // weight_vol …                    : input tile buffer (padded)
    extern __shared__ float shmem[];
    const int weight_vol = k * k * in_c;           // #floats per output channel
    float* sm_weight = shmem;                      // size = weight_vol
    float* sm_input  = shmem + weight_vol;         // size = (TILE+K-1)*(TILE+K-1)*in_c padded

    // -----------------------------------------------------------------
    //  1) Load weights of this output channel into shared memory
    // -----------------------------------------------------------------
    for (int idx = threadIdx.y * blockDim.x + threadIdx.x;
         idx < weight_vol;
         idx += blockDim.x * blockDim.y)
    {
        sm_weight[idx] = weight[oc * weight_vol + idx];
    }

    // -----------------------------------------------------------------
    //  2) Load the required input patch into shared memory
    // -----------------------------------------------------------------
    // Input patch size per channel:
    //   (TILE + K - 1) rows × (TILE + K - 1) cols
    const int patch_h = TILE + k - 1;
    const int patch_w = TILE + k - 1;
    // Pad each row by BANK_PAD floats to avoid bank conflicts
    const int pitch = patch_w + BANK_PAD;          // stride inside shared memory

    // Total number of elements we have to load (per channel)
    const int patch_vol = patch_h * pitch;        // note the padded pitch

    for (int ic = 0; ic < in_c; ++ic) {
        // Global pointer to the first element of the needed patch (top‑left)
        const int in_y0 = tile_row * TILE;        // upper‑left corner in input (no padding)
        const int in_x0 = tile_col * TILE;

        // Load the patch for this channel cooperatively
        for (int idx = threadIdx.y * blockDim.x + threadIdx.x;
             idx < patch_h * patch_w;
             idx += blockDim.x * blockDim.y)
        {
            const int py = idx / patch_w;          // row inside the patch
            const int px = idx % patch_w;          // col inside the patch

            const int iy = in_y0 + py;             // absolute input y
            const int ix = in_x0 + px;             // absolute input x

            // Guard against image boundaries (rare: only at the right / bottom edge)
            float val = 0.f;
            if (iy < in_h && ix < in_w) {
                // input layout: N C H W
                const int in_idx = ((b * in_c + ic) * in_h + iy) * in_w + ix;
                val = input[in_idx];
            }

            // Store into shared memory (with row padding)
            sm_input[ic * patch_vol + py * pitch + px] = val;
        }
    }

    __syncthreads();   // make sure all data is available

    // -----------------------------------------------------------------
    //  3) Compute the convolution (if inside the valid output region)
    // -----------------------------------------------------------------
    if (out_y < out_h && out_x < out_w) {
        float acc = bias[oc];                     // start from bias

        // Convolution over K×K kernel and all input channels
        for (int ky = 0; ky < k; ++ky) {
            const int py = ty + ky;               // position inside shared input patch
            for (int kx = 0; kx < k; ++kx) {
                const int px = tx + kx;           // position inside shared input patch

                // accumulate over input channels
                for (int ic = 0; ic < in_c; ++ic) {
                    const float in_val = sm_input[ic * patch_vol + py * pitch + px];
                    const float w_val  = sm_weight[(ky * k + kx) * in_c + ic];
                    acc += in_val * w_val;
                }
            }
        }

        // -----------------------------------------------------------------
        //  4) Mish activation  (y = x * tanh(softplus(x)))
        // -----------------------------------------------------------------
        float val = acc - sub1 - sub2;
        float mish = val * tanhf(logf(1.0f + expf(val)));
        const int out_idx = ((b * out_c + oc) * out_h + out_y) * out_w + out_x;
        output[out_idx] = mish;
    }
}

// ---------------------------------------------------------------------
//  Host wrapper (called from Python)
// ---------------------------------------------------------------------
void fused_conv_mish(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float sub1,
    float sub2)
{
    const int batch  = input.size(0);
    const int in_c   = input.size(1);
    const int in_h   = input.size(2);
    const int in_w   = input.size(3);
    const int out_c  = weight.size(0);
    const int k      = weight.size(1);                 // kernel height == width
    const int out_h  = in_h - k + 1;
    const int out_w  = in_w - k + 1;

    const dim3 threads(TILE, TILE);
    const dim3 blocks(
        ((out_h + TILE - 1) / TILE) * ((out_w + TILE - 1) / TILE),   // tile grid
        out_c,
        batch);

    // shared memory: weights + input tile (padded)
    const int weight_vol = k * k * in_c;
    const int patch_h    = TILE + k - 1;
    const int patch_w    = TILE + k - 1;
    const int pitch      = patch_w + 1;               // BANK_PAD = 1
    const int patch_vol  = patch_h * pitch;           // per‑channel volume with padding
    const size_t shmem_bytes = (weight_vol + in_c * patch_vol) * sizeof(float);

    fused_conv_mish_kernel<<<blocks, threads, shmem_bytes>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_c, in_h, in_w,
        out_c, k, out_h, out_w,
        sub1, sub2);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_mish(torch::Tensor input,
                    torch::Tensor weight,
                    torch::Tensor bias,
                    torch::Tensor output,
                    float sub1,
                    float sub2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &fused_conv_mish, "Fused Conv + Mish (shared‑mem input)");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# ---------------------------------------------------------------------
#  Python‑level functional model (unchanged API)
# ---------------------------------------------------------------------
def functional_model(x, *, conv_weight, conv_bias,
                    conv_stride=1, conv_padding=0,
                    conv_dilation=1, conv_groups=1,
                    subtract_value_1, subtract_value_2):
    """
    Executes the fused convolution + Mish activation using the custom CUDA kernel.
    The signature mimics a regular torch.nn.Conv2d forward pass so that the test
    harness can call it directly.
    """
    # The kernel expects weights in [out_c, k, k, in_c] order (flattened)
    w_reordered = conv_weight.permute(0, 2, 3, 1).contiguous()

    batch, _, h, w = x.shape
    k = conv_weight.shape[2]           # kernel height (== width)
    out_h = h - k + 1
    out_w = w - k + 1

    out = torch.empty((batch, conv_weight.size(0), out_h, out_w),
                      dtype=x.dtype, device=x.device)

    fused_ext.fused_conv(
        x, w_reordered, conv_bias, out,
        subtract_value_1, subtract_value_2)

    return out
