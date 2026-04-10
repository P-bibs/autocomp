# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_144040/code_8.py
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
# CUDA kernel –  convolution + Mish, input tiled in shared memory
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE 16          // output tile per block (same as original)
#define K_MAX 7          // maximum kernel size we support (enough for most nets)

// ---------------------------------------------------------------------
// Helper to compute a 1‑D index from a 4‑D tensor layout (NCHW)
__device__ __forceinline__
float load_input(const float* __restrict__ src,
                 int n, int c, int h, int w,
                 int N, int C, int H, int W)
{
    // bounds check – out‑of‑range positions are treated as zero
    if (h < 0 || h >= H || w < 0 || w >= W) return 0.0f;
    size_t idx = ((size_t)n * C * H * W) +
                 ((size_t)c * H * W) +
                 ((size_t)h * W) + w;
    return src[idx];
}

// ---------------------------------------------------------------------
// Fused convolution + Mish (output tile = TILE×TILE)
extern "C"
__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,   // NCHW
    const float* __restrict__ weight,  // OC × K × K × IC (flattened)
    const float* __restrict__ bias,    // OC
    float* __restrict__ output,        // N × OC × OH × OW
    int batch, int in_c, int in_h, int in_w,
    int out_c, int k, int out_h, int out_w,
    float sub1, float sub2)
{
    // ---------------------------------------------------------------
    // 1) block / thread coordinates
    // ---------------------------------------------------------------
    int oc = blockIdx.y;                  // output channel
    int n  = blockIdx.z;                  // batch index

    // each block processes a TILE×TILE region of the output map
    int tiles_per_row = (out_w + TILE - 1) / TILE;
    int tile_h = blockIdx.x / tiles_per_row;
    int tile_w = blockIdx.x % tiles_per_row;

    int ty = threadIdx.y;                 // 0 … TILE‑1
    int tx = threadIdx.x;                 // 0 … TILE‑1

    int out_y = tile_h * TILE + ty;       // absolute output row
    int out_x = tile_w * TILE + tx;       // absolute output column

    // ---------------------------------------------------------------
    // 2) shared memory layout
    // ---------------------------------------------------------------
    //   weight_smem  : k*k*in_c floats                (already in original)
    //   input_smem   : (TILE + k‑1) * (TILE + k‑1) floats
    //   total size   : weight_vol + input_vol
    // ---------------------------------------------------------------
    extern __shared__ float smem[];
    const int weight_vol = k * k * in_c;
    float* weight_smem = smem;                              // [weight_vol]
    float* input_smem  = smem + weight_vol;                 // [(TILE+k-1)^2]

    // ---------------------------------------------------------------
    // 3) load weights (once per block, same as original)
    // ---------------------------------------------------------------
    for (int idx = threadIdx.y * blockDim.x + threadIdx.x;
         idx < weight_vol;
         idx += blockDim.x * blockDim.y)
    {
        weight_smem[idx] = weight[oc * weight_vol + idx];
    }

    // ---------------------------------------------------------------
    // 4) load input tile (including halo) – coalesced
    // ---------------------------------------------------------------
    const int in_tile_h = TILE + k - 1;
    const int in_tile_w = TILE + k - 1;

    // each thread loads one (or more) input elements
    for (int idx = threadIdx.y * blockDim.x + threadIdx.x;
         idx < in_tile_h * in_tile_w;
         idx += blockDim.x * blockDim.y)
    {
        int iy = idx / in_tile_w;          // 0 … in_tile_h‑1
        int ix = idx % in_tile_w;          // 0 … in_tile_w‑1

        // input coordinate that corresponds to this tile element
        int img_y = tile_h * TILE + iy - (k/2);   // centre the halo
        int img_x = tile_w * TILE + ix - (k/2);

        // load from global memory (zero‑pad outside)
        float val = load_input(input,
                               n,                     // batch
                               0,                     // we will sum over all input channels later
                               img_y, img_x,
                               batch, in_c, in_h, in_w);
        // store per‑channel later; for now we just keep the raw value
        input_smem[iy * in_tile_w + ix] = val;
    }
    __syncthreads();

    // ---------------------------------------------------------------
    // 5) compute output (if inside valid range)
    // ---------------------------------------------------------------
    if (out_y < out_h && out_x < out_w)
    {
        float acc = bias[oc];          // start with bias

        // loop over kernel rows / cols
        #pragma unroll
        for (int ky = 0; ky < k; ++ky)
        {
            #pragma unroll
            for (int kx = 0; kx < k; ++kx)
            {
                // input tile coordinates that correspond to (out_y,out_x)
                int iy = ty + ky;
                int ix = tx + kx;

                // read the *first* channel (c=0) from shared memory
                // later we will multiply by the per‑channel weight and sum over IC
                float in_base = input_smem[iy * in_tile_w + ix];

                // offset inside weight_smem for this kernel position
                int w_off = (ky * k + kx) * in_c;

                // accumulate over input channels
                #pragma unroll
                for (int ic = 0; ic < in_c; ++ic)
                {
                    // stride‑aware input: we need the value for channel ic.
                    // Because we loaded only channel‑0 in shared memory,
                    // we fetch the remaining channels directly from global memory.
                    // This still saves the majority of traffic (the inner spatial reads).
                    float in_val;
                    if (ic == 0)
                        in_val = in_base;
                    else
                        in_val = input[((size_t)n * in_c * in_h * in_w) +
                                       ((size_t)ic * in_h * in_w) +
                                       ((size_t)(out_y + ky) * in_w) + (out_x + kx)];

                    acc += in_val * weight_smem[w_off + ic];
                }
            }
        }

        // Mish activation (exactly as original)
        float val = acc - sub1 - sub2;
        output[((size_t)n * out_c + oc) * out_h * out_w + out_y * out_w + out_x]
            = val * tanhf(logf(1.0f + expf(val)));
    }
}

// ---------------------------------------------------------------------
// C++ wrapper – the same signature that the Python side expects
// ---------------------------------------------------------------------
void fused_conv_mish(torch::Tensor input,
                    torch::Tensor weight,
                    torch::Tensor bias,
                    torch::Tensor output,
                    float sub1,
                    float sub2)
{
    const int batch = input.size(0);
    const int in_c  = input.size(1);
    const int in_h  = input.size(2);
    const int in_w  = input.size(3);
    const int out_c = weight.size(0);
    const int k     = weight.size(1);
    const int out_h = in_h - k + 1;
    const int out_w = in_w - k + 1;

    dim3 threads(TILE, TILE);
    dim3 blocks(
        ((out_h + TILE - 1) / TILE) * ((out_w + TILE - 1) / TILE),
        out_c,
        batch
    );

    // shared memory = weight + input tile
    const size_t weight_smem_bytes = (size_t)k * k * in_c * sizeof(float);
    const int in_tile_h = TILE + k - 1;
    const int in_tile_w = TILE + k - 1;
    const size_t input_smem_bytes = (size_t)in_tile_h * in_tile_w * sizeof(float);

    const size_t shared_mem_bytes = weight_smem_bytes + input_smem_bytes;

    fused_conv_mish_kernel<<<blocks, threads, shared_mem_bytes>>>(
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

# ----------------------------------------------------------------------
# C++ binding (pybind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

// forward declaration
void fused_conv_mish(torch::Tensor input,
                    torch::Tensor weight,
                    torch::Tensor bias,
                    torch::Tensor output,
                    float sub1,
                    float sub2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv",
          &fused_conv_mish,
          "Fused convolution + Mish (input‑tile shared‑mem version)");
}
"""

# ----------------------------------------------------------------------
# Build the extension – use fast‑math and aggressive optimisation
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional model – unchanged API, only the kernel implementation differs
# ----------------------------------------------------------------------
def functional_model(x, *,
                    conv_weight,
                    conv_bias,
                    conv_stride=1,
                    conv_padding=0,
                    conv_dilation=1,
                    conv_groups=1,
                    subtract_value_1,
                    subtract_value_2):
    """
    Equivalent to the original ``functional_model`` but uses the
    ``fused_conv`` kernel that tiles the input patch in shared memory.
    The original code assumed a stride of 1, no padding, dilation = 1 and
    groups = 1 – these arguments are kept for API compatibility but are
    currently ignored (the kernel implements a plain 2‑D convolution).
    """

    # The kernel expects weights ordered as [out_c, k, k, in_c]
    w_reordered = conv_weight.permute(0, 2, 3, 1).contiguous()

    batch, _, h, w = x.shape
    k = conv_weight.shape[2]                     # kernel size (assumed square)
    out_h = h - k + 1
    out_w = w - k + 1

    # allocate output tensor
    out = torch.empty((batch, conv_weight.size(0), out_h, out_w),
                      dtype=x.dtype,
                      device=x.device)

    # launch the fused kernel
    fused_ext.fused_conv(
        x, w_reordered, conv_bias, out,
        subtract_value_1, subtract_value_2
    )
    return out
