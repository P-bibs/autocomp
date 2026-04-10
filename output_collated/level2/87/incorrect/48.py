# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150347/code_8.py
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
# CUDA kernel: input-tile shared-memory caching + weight caching
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE 16                        // output tile per block (must be multiple of 32)

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int in_c, int in_h, int in_w,
    int out_c, int k, int out_h, int out_w,
    float sub1, float sub2)
{
    // -----------------------------------------------------------------
    // Block identifies a tile of the output for a given (output channel, batch)
    // -----------------------------------------------------------------
    const int oc = blockIdx.y;            // output channel
    const int b  = blockIdx.z;            // batch index

    // -----------------------------------------------------------------
    // Shared memory layout:
    //   [weight (k*k*in_c)]  followed by  [input tile (in_c * (TILE+k-1) * (TILE+k-1))]
    // -----------------------------------------------------------------
    extern __shared__ float smem[];
    float* weight_smem = smem;                         // size = k*k*in_c
    const int weight_vol = k * k * in_c;

    // -------------------------------------------------------------
    // 1) Load weights for this output channel into shared memory
    // -------------------------------------------------------------
    for (int idx = threadIdx.y * blockDim.x + threadIdx.x;
         idx < weight_vol;
         idx += blockDim.x * blockDim.y) {
        weight_smem[idx] = weight[oc * weight_vol + idx];
    }

    // -------------------------------------------------------------
    // 2) Load input tile (including halo) into shared memory
    // -------------------------------------------------------------
    const int tile_h = TILE + k - 1;   // height of input tile needed for this output tile
    const int tile_w = TILE + k - 1;   // width  of input tile needed for this output tile
    float* in_tile = weight_smem + weight_vol;   // start of input-tile buffer

    // top-left corner of the output tile in output coordinates
    const int out_tile_x = blockIdx.x * TILE;   // output column start for this block
    const int out_tile_y = threadIdx.y;         // will be used as base for loading rows

    // Load the whole (in_c, tile_h, tile_w) region cooperatively.
    // Each thread may load several (iy,ix) positions.
    for (int iy = threadIdx.y; iy < tile_h; iy += blockDim.y) {
        for (int ix = threadIdx.x; ix < tile_w; ix += blockDim.x) {
            int in_y = out_tile_y + iy;   // corresponding input row
            int in_x = out_tile_x + ix;   // corresponding input column

            // Bounds check – outside image => zero (same as implicit zero padding in the original impl)
            bool inside = (in_y < in_h) && (in_x < in_w);
            for (int ic = 0; ic < in_c; ++ic) {
                float val = 0.0f;
                if (inside) {
                    // input layout: (b, C, H, W)
                    int idx = ((b * in_c + ic) * in_h + in_y) * in_w + in_x;
                    val = input[idx];
                }
                // store as [ic][iy][ix] (contiguous in ix)
                in_tile[(ic * tile_h + iy) * tile_w + ix] = val;
            }
        }
    }
    __syncthreads();

    // -----------------------------------------------------------------
    // 3) Compute output elements inside this TILE×TILE region
    // -----------------------------------------------------------------
    const int stride = blockDim.x * blockDim.y;
    const int tid    = threadIdx.y * blockDim.x + threadIdx.x;
    const int outputs_per_tile = TILE * TILE;

    for (int linear = tid; linear < outputs_per_tile; linear += stride) {
        int ox = linear % TILE;           // column inside tile
        int oy = linear / TILE;           // row    inside tile
        int out_x = out_tile_x + ox;
        int out_y = out_tile_y + oy;

        // guard against tiles that exceed the actual output size
        if (out_x >= out_w || out_y >= out_h) continue;

        float acc = bias[oc];             // start from bias

        // ----- convolution using data from shared memory -----
        for (int i = 0; i < k; ++i) {
            int iy = oy + i;              // index inside input tile
            for (int j = 0; j < k; ++j) {
                int ix = ox + j;          // index inside input tile
                #pragma unroll
                for (int ic = 0; ic < in_c; ++ic) {
                    float in_val = in_tile[(ic * tile_h + iy) * tile_w + ix];
                    float w_val  = weight_smem[(i * k + j) * in_c + ic];
                    acc += in_val * w_val;
                }
            }
        }

        // ----- Mish activation (identical to original implementation) -----
        float val  = acc - sub1 - sub2;
        float mish = val * tanhf(logf(1.0f + expf(val)));

        // store result
        output[((b * out_c + oc) * out_h + out_y) * out_w + out_x] = mish;
    }
}

// -------------------------------------------------------------------------
// Host wrapper
// -------------------------------------------------------------------------
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
    const int k     = weight.size(1);
    const int out_h = in_h - k + 1;
    const int out_w = in_w - k + 1;

    // Block = TILE × TILE threads (256 = 16×16, a multiple of 32)
    dim3 threads(TILE, TILE);

    // Grid: one block per output channel, per batch, and per horizontal tile
    const int tiles_x = (out_w + TILE - 1) / TILE;
    dim3 blocks(tiles_x, out_c, batch);

    // Shared-memory size = weight buffer + input-tile buffer
    const size_t weight_smem_bytes = k * k * in_c * sizeof(float);
    const size_t in_tile_bytes    = in_c * (TILE + k - 1) * (TILE + k - 1) * sizeof(float);
    const size_t shared_bytes      = weight_smem_bytes + in_tile_bytes;

    fused_conv_mish_kernel<<<blocks, threads, shared_bytes>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_c, in_h, in_w,
        out_c, k, out_h, out_w,
        sub1, sub2);
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_mish(torch::Tensor input,
                     torch::Tensor weight,
                     torch::Tensor bias,
                     torch::Tensor output,
                     float sub1,
                     float sub2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &fused_conv_mish,
          "Fused convolution + Mish activation with input-tile caching (CUDA)");
}
"""

# -------------------------------------------------------------------------
# Compile the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional model exposed to the outside world
# -------------------------------------------------------------------------
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
    Forward pass that uses the custom fused CUDA kernel.
    Expected layout for ``conv_weight`` is the standard PyTorch
    (out_channels, in_channels, k, k).  The kernel requires the
    layout (out_channels, k, k, in_channels) which we obtain by a
    permutation.
    """
    # Re-order weights to match kernel's memory layout
    w_reordered = conv_weight.permute(0, 2, 3, 1).contiguous()

    batch, _, h, w = x.shape
    k = conv_weight.shape[2]                     # kernel size (assumed square)
    out_h = h - k + 1
    out_w = w - k + 1

    # Allocate output tensor
    out = torch.empty((batch,
                       conv_weight.size(0),
                       out_h,
                       out_w),
                      device=x.device,
                      dtype=x.dtype)

    # Invoke the fused kernel
    fused_ext.fused_conv(x,
                         w_reordered,
                         conv_bias,
                         out,
                         float(subtract_value_1),
                         float(subtract_value_2))

    return out
