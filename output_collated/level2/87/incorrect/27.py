# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_142828/code_14.py
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
# CUDA source – tiled convolution with shared-memory weight & input caches
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_H 4   // output tile height
#define TILE_W 4   // output tile width

// ------------------------------------------------------------------------
// Kernel: 4x4 output tile, per-block weight & input caches, Mish activation
// ------------------------------------------------------------------------
__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_c,
    const int in_h,
    const int in_w,
    const int out_c,
    const int k,
    const float sub1,
    const float sub2,
    const int tile_h,
    const int tile_w)
{
    // ---- shared memory layout ------------------------------------------------
    // weightCache  : in_c * k * k  floats
    // inputCache   : in_c * (tile_h+k-1) * (tile_w+k-1) floats
    extern __shared__ float smem[];
    float* weightCache = smem;
    const int weightSize = in_c * k * k;
    float* inputCache   = smem + weightSize;
    const int tile_i_h = tile_h + k - 1;
    const int tile_i_w = tile_w + k - 1;
    const int inputTileSize = in_c * tile_i_h * tile_i_w;

    // ---- block decomposition -------------------------------------------------
    // grid.x = batch * out_c
    // grid.y = number of vertical tiles
    // grid.z = number of horizontal tiles
    const int blockId   = blockIdx.x;
    const int b         = blockId / out_c;          // batch index
    const int oc        = blockId % out_c;          // output-channel index
    const int tileRow   = blockIdx.y;
    const int tileCol   = blockIdx.z;

    const int startOh   = tileRow * tile_h;
    const int startOw   = tileCol * tile_w;

    const int oh = startOh + threadIdx.y;   // output row (0..tile_h-1)
    const int ow = startOw + threadIdx.x;   // output col (0..tile_w-1)

    const int out_h = in_h - k + 1;
    const int out_w = in_w - k + 1;

    // out-of-bounds guard
    if (oh >= out_h || ow >= out_w) return;

    // ---- base offsets --------------------------------------------------------
    const long weightBase = ((long)oc * in_c) * k * k;
    const long inputBase  = ((long)b * in_c) * in_h * in_w;

    // ---- load weight cache (once per block) ---------------------------------
    const int nThreads = blockDim.x * blockDim.y;   // = TILE_H * TILE_W = 16
    for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
         idx < weightSize;
         idx += nThreads)
    {
        weightCache[idx] = __ldg(&weight[weightBase + idx]);
    }

    // ---- load input tile (coalesced) ----------------------------------------
    for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
         idx < inputTileSize;
         idx += nThreads)
    {
        const int ic = idx / (tile_i_h * tile_i_w);
        const int rem = idx % (tile_i_h * tile_i_w);
        const int r = rem / tile_i_w;          // local row inside the tile
        const int c = rem % tile_i_w;          // local col inside the tile
        const int gR = startOh + r;            // global row in the full input
        const int gC = startOw + c;            // global col in the full input
        const long iIdx = ((long)ic * in_h + gR) * in_w + gC;
        // Handle boundary conditions
        if (gR < in_h && gC < in_w) {
            inputCache[idx] = __ldg(&input[inputBase + iIdx]);
        } else {
            inputCache[idx] = 0.0f;
        }
    }

    __syncthreads();

    // ---- convolution ---------------------------------------------------------
    float acc = bias[oc];

    for (int ic = 0; ic < in_c; ++ic)
    {
        for (int ki = 0; ki < k; ++ki)
        {
            #pragma unroll
            for (int kj = 0; kj < k; ++kj)
            {
                const int wIdx = ((ic * k) + ki) * k + kj;
                const float wVal = weightCache[wIdx];

                const int rowOff = oh + ki - startOh;
                const int colOff = ow + kj - startOw;
                const int inIdx = ((ic * tile_i_h) + rowOff) * tile_i_w + colOff;
                const float inVal = inputCache[inIdx];

                acc += inVal * wVal;
            }
        }
    }

    // ---- subtract constants & Mish -----------------------------------------
    float val = acc - sub1 - sub2;
    output[(((((long)b * out_c + oc) * out_h + oh) * out_w) + ow)] =
        val * tanhf(logf(1.0f + expf(val)));
}

// ------------------------------------------------------------------------
// Host wrapper that sets up the grid and launches the kernel
// ------------------------------------------------------------------------
void fused_conv_mish(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float sub1,
    float sub2)
{
    const int batch   = input.size(0);
    const int in_c    = input.size(1);
    const int in_h    = input.size(2);
    const int in_w    = input.size(3);
    const int out_c   = weight.size(0);
    const int k       = weight.size(2);
    const int out_h   = in_h - k + 1;
    const int out_w   = in_w - k + 1;

    const int tile_h = TILE_H;
    const int tile_w = TILE_W;

    const int ntile_h = (out_h + tile_h - 1) / tile_h;
    const int ntile_w = (out_w + tile_w - 1) / tile_w;

    const int gridX = batch * out_c;
    const int gridY = ntile_h;
    const int gridZ = ntile_w;

    dim3 grid(gridX, gridY, gridZ);
    dim3 block(tile_w, tile_h);   // 4×4 = 16 threads per block

    const int weightSize   = in_c * k * k;
    const int tile_i_h     = tile_h + k - 1;
    const int tile_i_w     = tile_w + k - 1;
    const int inputTileSize = in_c * tile_i_h * tile_i_w;
    const size_t sharedMem = (weightSize + inputTileSize) * sizeof(float);

    fused_conv_mish_kernel<<<grid, block, sharedMem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_c, in_h, in_w,
        out_c, k, sub1, sub2,
        tile_h, tile_w);
}
"""

# ----------------------------------------------------------------------
# C++ binding (PYBIND11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_mish(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float sub1,
    float sub2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_mish", &fused_conv_mish,
          "Fused convolution, subtraction and Mish (CUDA)");
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
# Functional model – entry point used during evaluation
# ----------------------------------------------------------------------
def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride=1,
    conv_padding=0,
    conv_dilation=1,
    conv_groups=1,
    subtract_value_1,
    subtract_value_2):
    """
    Expected shapes:
        x         : (batch, in_c, in_h, in_w)
        conv_weight: (out_c, in_c, k, k)
        conv_bias  : (out_c,)
    The kernel assumes stride=1, padding=0, dilation=1, groups=1.
    """
    batch, in_c, in_h, in_w = x.shape
    k = conv_weight.shape[2]                # kernel size (square)
    out_c = conv_weight.shape[0]
    out_h = in_h - k + 1
    out_w = in_w - k + 1

    # allocate output tensor
    out = torch.empty((batch, out_c, out_h, out_w),
                      device=x.device, dtype=x.dtype)

    fused_ext.fused_conv_mish(
        x, conv_weight, conv_bias, out,
        subtract_value_1, subtract_value_2)

    return out
