# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_144040/code_10.py
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
# CUDA kernel – tiled convolution with shared-memory input reuse
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#ifndef TILE_DIM
#define TILE_DIM 16          // 16×16 output tile per block
#endif
#ifndef CHUNK
#define CHUNK   8           // channels per shared-memory chunk
#endif

// ------------------------------------------------------------------
// Kernel
// ------------------------------------------------------------------
__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int in_c, int in_h, int in_w,
    int out_c, int k,
    float sub1, float sub2,
    int out_h, int out_w)
{
    // ------------------------------------------------------------------
    // 1) Identify the (batch, out_channel) pair handled by this block
    // ------------------------------------------------------------------
    const int bo = blockIdx.z;                 // 0 … batch*out_c‑1
    const int b  = bo / out_c;                 // batch index
    const int oc = bo % out_c;                 // output channel

    // ------------------------------------------------------------------
    // 2) Tile origin in the output (top‑left corner of this block)
    // ------------------------------------------------------------------
    const int tile_x = blockIdx.x * TILE_DIM;  // output width coordinate
    const int tile_y = blockIdx.y * TILE_DIM;  // output height coordinate

    // ------------------------------------------------------------------
    // 3) Thread‑local output coordinates (one pixel per thread)
    // ------------------------------------------------------------------
    const int ox = tile_x + threadIdx.x;       // output x
    const int oy = tile_y + threadIdx.y;       // output y

    // Early‑exit for threads that fall outside the valid output region
    if (ox >= out_w || oy >= out_h) return;

    // ------------------------------------------------------------------
    // 4) Accumulator (starts with bias)
    // ------------------------------------------------------------------
    float acc = bias[oc];

    // ------------------------------------------------------------------
    // 5) Shared memory: holds a CHUNK‑sized slice of the input tile
    // ------------------------------------------------------------------
    extern __shared__ float shmem[];            // size = CHUNK * (TILE_DIM+K-1)^2
    const int sh_h = TILE_DIM + k - 1;          // height of the input tile
    const int sh_w = TILE_DIM + k - 1;          // width  of the input tile
    const int sh_slice = sh_h * sh_w;           // elements per channel

    // ------------------------------------------------------------------
    // 6) Loop over input channels in CHUNK‑sized chunks
    // ------------------------------------------------------------------
    for (int c0 = 0; c0 < in_c; c0 += CHUNK) {
        const int cur_c = min(CHUNK, in_c - c0);   // actual channels in this chunk

        // --------------------------------------------------------------
        // 6.1) Load the required input region for the current chunk
        // --------------------------------------------------------------
        for (int ic = 0; ic < cur_c; ++ic) {
            const int g_ic = c0 + ic;               // global input channel index
            // each thread loads several positions (stride over blockDim)
            for (int dy = threadIdx.y; dy < sh_h; dy += blockDim.y) {
                const int in_y = tile_y + dy;        // absolute input y
                for (int dx = threadIdx.x; dx < sh_w; dx += blockDim.x) {
                    const int in_x = tile_x + dx;    // absolute input x
                    float v = 0.0f;
                    if (in_y < in_h && in_x < in_w) {
                        // input layout: [B, C, H, W] (row‑major)
                        const size_t idx = ((size_t)b * in_c + g_ic) * in_h * in_w
                                           + (size_t)in_y * in_w + in_x;
                        v = input[idx];
                    }
                    shmem[ic * sh_slice + dy * sh_w + dx] = v;
                }
            }
        }
        __syncthreads();

        // --------------------------------------------------------------
        // 6.2) Compute the contribution of this chunk to the output pixel
        // --------------------------------------------------------------
        for (int ic = 0; ic < cur_c; ++ic) {
            const int g_ic = c0 + ic;
            // weight layout (already reordered): [out_c, K, K, in_c]
            const float* w_ptr = weight + ((size_t)oc * k * k * in_c)
                                 + (size_t)g_ic;   // start of this channel's weights
            for (int i = 0; i < k; ++i) {
                for (int j = 0; j < k; ++j) {
                    const float in_val = shmem[ic * sh_slice
                                      + (threadIdx.y + i) * sh_w
                                      + (threadIdx.x + j)];
                    // weight index: ((i*k + j) * in_c + g_ic)
                    const float w_val = w_ptr[(i * k + j) * in_c];
                    acc += in_val * w_val;
                }
            }
        }
        __syncthreads();   // before loading the next chunk
    }

    // ------------------------------------------------------------------
    // 7) Mish activation (still fused with the two subtractions)
    // ------------------------------------------------------------------
    float val = acc - sub1 - sub2;
    float mish = val * tanhf(logf(1.0f + expf(val)));

    // ------------------------------------------------------------------
    // 8) Write the result
    // ------------------------------------------------------------------
    const size_t out_idx = ((size_t)b * out_c + oc) * out_h * out_w
                         + (size_t)oy * out_w + ox;
    output[out_idx] = mish;
}

// ----------------------------------------------------------------------
// Host wrapper – computes grid/block dimensions and launches the kernel
// ----------------------------------------------------------------------
void fused_conv_mish(
    torch::Tensor input,
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
    const int k     = weight.size(2);               // kernel size (assumed square)

    const int out_h = in_h - k + 1;
    const int out_w = in_w - k + 1;

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid( (out_w + TILE_DIM - 1) / TILE_DIM,
               (out_h + TILE_DIM - 1) / TILE_DIM,
               (size_t)batch * out_c );

    // shared memory size for ONE chunk (CHUNK channels)
    const size_t sh_bytes = (size_t)CHUNK * (TILE_DIM + k - 1) *
                            (TILE_DIM + k - 1) * sizeof(float);

    fused_conv_mish_kernel<<<grid, block, sh_bytes>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_c, in_h, in_w,
        out_c, k,
        sub1, sub2,
        out_h, out_w);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA launch error: %s\n", cudaGetErrorString(err));
    }
}
"""

# ----------------------------------------------------------------------
# C++ bindings (PyBind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the CUDA implementation
void fused_conv_mish(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float sub1,
    float sub2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &fused_conv_mish,
          "Fused conv + Mish (shared‑memory version)");
}
"""

# ----------------------------------------------------------------------
# Compile the extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Public API – exactly the same signature as the original functional_model
# ----------------------------------------------------------------------
def functional_model(x, *, conv_weight, conv_bias,
                    conv_stride=1, conv_padding=0,
                    conv_dilation=1, conv_groups=1,
                    subtract_value_1, subtract_value_2):
    """
    x                : [B, C_in, H, W]  torch.float32 on CUDA
    conv_weight      : [C_out, C_in, K, K]   (will be reordered)
    conv_bias        : [C_out]
    subtract_value_1, subtract_value_2 : scalars
    """
    # 1) Reorder weights to the layout expected by the kernel:
    #    [out_c, K, K, in_c] (i.e. NCHW → N H W C)
    w_reordered = conv_weight.permute(0, 2, 3, 1).contiguous()

    B, _, H, W = x.shape
    K = conv_weight.shape[2]
    out_h = H - K + 1
    out_w = W - K + 1

    out = torch.empty((B, conv_weight.size(0), out_h, out_w),
                      dtype=x.dtype, device=x.device)

    fused_ext.fused_conv(
        x, w_reordered, conv_bias, out,
        subtract_value_1, subtract_value_2
    )
    return out
