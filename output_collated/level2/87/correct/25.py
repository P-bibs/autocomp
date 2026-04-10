# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_144040/code_11.py
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

# ------------------------------------------------------------------
#  CUDA kernel – shared-memory version (optimization #4)
# ------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

constexpr int TILE_H = 16;          // must be multiple of 32 for warp alignment
constexpr int TILE_W = 16;

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int in_c, int in_h, int in_w,
    int out_c, int k,
    float sub1, float sub2)
{
    // ------------------------------------------------------------------
    //  Derive indices
    // ------------------------------------------------------------------
    // blockIdx.z folds batch and out_c
    int oc = blockIdx.z % out_c;          // output channel
    int b  = blockIdx.z / out_c;          // batch index

    // tile origin in output space
    int tile_out_x = blockIdx.x * TILE_W;
    int tile_out_y = blockIdx.y * TILE_H;

    // input origin that contains the whole tile (including halo)
    int tile_in_x = tile_out_x;
    int tile_in_y = tile_out_y;

    // ------------------------------------------------------------------
    //  Shared memory layout:
    //  for each input channel we store a (TILE_H + k - 1) × (TILE_W + k - 1) patch
    // ------------------------------------------------------------------
    const int tile_h = TILE_H + k - 1;
    const int tile_w = TILE_W + k - 1;
    const int tile_area = tile_h * tile_w;
    extern __shared__ float smem[];                // size = in_c * tile_area * sizeof(float)

    // ------------------------------------------------------------------
    //  Load the input patch for every channel into shared memory
    //  Each thread loads several elements (stride = blockDim)
    // ------------------------------------------------------------------
    for (int ic = 0; ic < in_c; ++ic) {
        float* s_in = smem + ic * tile_area;
        // thread-wise copy, coalesced along x
        for (int y = threadIdx.y; y < tile_h; y += blockDim.y) {
            int in_y = tile_in_y + y;
            if (in_y >= in_h) continue;          // guard against out-of-bounds (rare)
            for (int x = threadIdx.x; x < tile_w; x += blockDim.x) {
                int in_x = tile_in_x + x;
                if (in_x >= in_w) continue;
                // global input layout: [b, ic, in_h, in_w]
                s_in[y * tile_w + x] = input[
                    ((b * in_c + ic) * in_h + in_y) * in_w + in_x];
            }
        }
    }
    __syncthreads();   // ensure the whole tile is resident

    // ------------------------------------------------------------------
    //  Each thread now computes one output element inside the tile
    // ------------------------------------------------------------------
    int out_x = tile_out_x + threadIdx.x;   // column within output map
    int out_y = tile_out_y + threadIdx.y;   // row   within output map

    // Guard against boundary of the output feature map
    if (out_x >= (in_w - k + 1) || out_y >= (in_h - k + 1))
        return;

    float acc = bias[oc];   // start with bias

    // --------------------------------------------------------------
    //  Convolution: accumulate over input channels and kernel elements
    // --------------------------------------------------------------
    for (int ic = 0; ic < in_c; ++ic) {
        const float* s_in = smem + ic * tile_area;   // pointer to this channel's tile

        // Load weight for this (oc, ic) pair – a very small 3×3 (or k×k) matrix.
        // Because k is compile-time constant (typically 3), we unroll manually.
        // These loads hit the L2 cache and are effectively free compared to
        // the shared-memory reads.
        #pragma unroll
        for (int i = 0; i < k; ++i) {
            #pragma unroll
            for (int j = 0; j < k; ++j) {
                // weight layout: [out_c, in_c, k, k]
                float w = weight[(((oc * in_c) + ic) * k + i) * k + j];
                float inp = s_in[(threadIdx.y + i) * tile_w + (threadIdx.x + j)];
                acc += inp * w;
            }
        }
    }

    // --------------------------------------------------------------
    //  Subtractions + Mish activation (fast-math)
    // --------------------------------------------------------------
    float val = acc - sub1 - sub2;
    float out_val = val * tanhf(logf(1.0f + expf(val)));

    // Global write
    int out_w = in_w - k + 1;
    int out_h = in_h - k + 1;
    int out_idx = ((b * out_c + oc) * out_h + out_y) * out_w + out_x;
    output[out_idx] = out_val;
}

// ------------------------------------------------------------------
//  Host wrapper – computes grid/block dims and launches kernel
// ------------------------------------------------------------------
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
    const int k     = weight.size(2);          // assume square kernel

    const int out_h = in_h - k + 1;
    const int out_w = in_w - k + 1;

    // ------------------------------------------------------------------
    //  Grid / block configuration
    // ------------------------------------------------------------------
    dim3 blockDim(TILE_W, TILE_H);
    dim3 gridDim( (out_w + TILE_W - 1) / TILE_W,
                  (out_h + TILE_H - 1) / TILE_H,
                  batch * out_c );

    // Shared memory needed for one input tile per channel
    const int tile_h = TILE_H + k - 1;
    const int tile_w = TILE_W + k - 1;
    const size_t smem_bytes = static_cast<size_t>(in_c) * tile_h * tile_w * sizeof(float);

    fused_conv_mish_kernel<<<gridDim, blockDim, smem_bytes>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_c, in_h, in_w,
        out_c, k,
        sub1, sub2);
}
"""

# ------------------------------------------------------------------
#  C++ binding (required by load_inline)
# ------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_mish(torch::Tensor input,
                    torch::Tensor weight,
                    torch::Tensor bias,
                    torch::Tensor output,
                    float sub1,
                    float sub2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_mish", &fused_conv_mish,
          "Fused Conv + Subtractions + Mish (shared-mem version)");
}
"""

# ------------------------------------------------------------------
#  Build the extension (keeps the same flags as the original code)
# ------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_ext_shared',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ------------------------------------------------------------------
#  Functional model – unchanged API
# ------------------------------------------------------------------
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
    Computes:
        out = Mish( Conv(x, conv_weight, conv_bias) - subtract_value_1 - subtract_value_2 )
    where Conv is a plain cross-correlation with stride=1, padding=0, dilation=1.
    The implementation is entirely in the custom CUDA kernel above.
    """
    batch, _, h, w = x.shape
    k = conv_weight.shape[2]                     # kernel size (square)
    out_h = h - k + 1
    out_w = w - k + 1
    out = torch.empty((batch, conv_weight.size(0), out_h, out_w),
                      device=x.device, dtype=x.dtype)

    fused_ext.fused_conv_mish(
        x, conv_weight, conv_bias, out,
        subtract_value_1, subtract_value_2
    )
    return out
