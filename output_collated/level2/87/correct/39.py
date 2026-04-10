# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_145317/code_12.py
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

# -------------------------------------------------------------------------
# Optimized CUDA kernel:  weight + input tile caching in shared memory,
# grid‑stride loop over spatial tiles, coalesced memory accesses.
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Tile size chosen to fit the combined weight+input tile into 48 KB of shared memory
#define TILE 8

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int in_c, int in_h, int in_w,
    int out_c, int k, int out_h, int out_w,
    float sub1, float sub2)
{
    // Shared memory layout:
    //   weight_smem : k*k*in_c  (cached kernel for the current output channel)
    //   input_smem  : (TILE+k-1)*(TILE+k-1)*in_c (cached input tile)
    extern __shared__ float smem[];
    float* weight_smem = smem;
    float* input_smem  = smem + k*k*in_c;

    const int tile_in_h = TILE + k - 1;
    const int tile_in_w = TILE + k - 1;
    const int input_vol = tile_in_h * tile_in_w * in_c;

    // Block indices
    const int oc          = blockIdx.y;               // output channel
    const int nb          = blockIdx.z;               // batch index
    const int nTilesX     = (out_w + TILE - 1) / TILE;
    const int nTilesY     = (out_h + TILE - 1) / TILE;
    const int totalTiles  = nTilesX * nTilesY;

    // -----------------------------------------------------------------
    // 1) Load weight for this output channel into shared memory (once)
    // -----------------------------------------------------------------
    const int weight_vol = k * k * in_c;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;
    for (int idx = tid; idx < weight_vol; idx += nthreads) {
        weight_smem[idx] = weight[oc * weight_vol + idx];
    }
    __syncthreads();

    // -----------------------------------------------------------------
    // 2) Process one or more spatial tiles with a grid‑stride loop
    // -----------------------------------------------------------------
    for (int tile_id = blockIdx.x; tile_id < totalTiles; tile_id += gridDim.x) {
        const int tile_y = tile_id / nTilesX;
        const int tile_x = tile_id % nTilesX;

        const int oh_base = tile_y * TILE;
        const int ow_base = tile_x * TILE;

        // ---- 2a) Cooperatively load the required input tile ----
        for (int idx = tid; idx < input_vol; idx += nthreads) {
            const int c   = idx / (tile_in_h * tile_in_w);
            const int rem = idx % (tile_in_h * tile_in_w);
            const int y   = rem / tile_in_w;
            const int x   = rem % tile_in_w;

            const int ih = oh_base + y;
            const int iw = ow_base + x;

            if (ih < in_h && iw < in_w) {
                // global memory access is coalesced because threads stride by nthreads
                input_smem[idx] = input[(nb * in_c * in_h * in_w) +
                                         (c * in_h * in_w) + (ih * in_w + iw)];
            } else {
                // zero‑pad out‑of‑bounds
                input_smem[idx] = 0.0f;
            }
        }
        __syncthreads();

        // ---- 2b) Compute output for the threads that fall inside the tile ----
        const int ty = threadIdx.y;
        const int tx = threadIdx.x;

        const int oh = oh_base + ty;
        const int ow = ow_base + tx;

        if (oh < out_h && ow < out_w) {
            float acc = bias[oc];

            // Convolution: iterate over kernel positions
            for (int ki = 0; ki < k; ++ki) {
                const int i_tile_y = ty + ki;            // y‑offset inside cached input tile
                for (int kj = 0; kj < k; ++kj) {
                    const int i_tile_x = tx + kj;        // x‑offset inside cached input tile
                    // innermost loop over input channels
                    for (int ic = 0; ic < in_c; ++ic) {
                        const float wVal = weight_smem[(ki * k + kj) * in_c + ic];
                        const float iVal = input_smem[((ic * tile_in_h) + i_tile_y) * tile_in_w + i_tile_x];
                        acc += wVal * iVal;
                    }
                }
            }

            // ---- 2c) Mish activation ----
            float val = acc - sub1 - sub2;
            float ex  = expf(val);
            float ln  = logf(1.0f + ex);
            float out_val = val * tanhf(ln);

            const int out_idx = ((nb * out_c + oc) * out_h + oh) * out_w + ow;
            output[out_idx] = out_val;
        }

        // Ensure all threads finish before loading the next tile
        __syncthreads();
    }
}

// Host wrapper
void fused_conv_mish(torch::Tensor input, torch::Tensor weight,
                     torch::Tensor bias, torch::Tensor output,
                     float sub1, float sub2) {
    const int batch   = input.size(0);
    const int in_c    = input.size(1);
    const int in_h    = input.size(2);
    const int in_w    = input.size(3);
    const int out_c   = weight.size(0);
    const int k       = weight.size(2);          // square kernel assumed
    const int out_h   = in_h - k + 1;
    const int out_w   = in_w - k + 1;

    const int nTilesX = (out_w + TILE - 1) / TILE;
    const int nTilesY = (out_h + TILE - 1) / TILE;
    const int totalTiles = nTilesX * nTilesY;

    dim3 threads(TILE, TILE);
    dim3 blocks(min(totalTiles, 65535), out_c, batch);

    const size_t weight_vol = k * k * in_c;
    const int tile_in_h = TILE + k - 1;
    const int tile_in_w = TILE + k - 1;
    const size_t input_vol = tile_in_h * tile_in_w * in_c;
    const size_t shared_mem = (weight_vol + input_vol) * sizeof(float);

    fused_conv_mish_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_c, in_h, in_w,
        out_c, k, out_h, out_w,
        sub1, sub2
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_conv_mish(torch::Tensor i, torch::Tensor w,
                     torch::Tensor b, torch::Tensor o,
                     float s1, float s2);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &fused_conv_mish,
          "Fused convolution followed by Mish activation (CUDA)");
}
"""

# -------------------------------------------------------------------------
# Build the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name="fused_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)

# -------------------------------------------------------------------------
# Functional wrapper that matches the required signature
# -------------------------------------------------------------------------
def functional_model(
    x: torch.Tensor,
    *,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_stride: int = 1,
    conv_padding: int = 0,
    conv_dilation: int = 1,
    conv_groups: int = 1,
    subtract_value_1: float,
    subtract_value_2: float,
) -> torch.Tensor:
    """
    Convolution (square kernel, no groups, stride=1, padding=0) followed by
    Mish activation.  All computation is performed by the custom CUDA kernel.
    """
    # Reorder weight from [out_c, in_c, k, k] -> [out_c, k, k, in_c] (contiguous)
    w = conv_weight.permute(0, 2, 3, 1).contiguous()

    batch, _, h, w_in = x.shape
    k = conv_weight.shape[2]  # square kernel size
    out_h = h - k + 1
    out_w = w_in - k + 1

    # Allocate output tensor
    out = torch.empty(
        (batch, conv_weight.size(0), out_h, out_w),
        device=x.device,
        dtype=x.dtype,
    )

    # Launch the fused kernel
    fused_ext.fused_conv(x, w, conv_bias, out, subtract_value_1, subtract_value_2)

    return out
