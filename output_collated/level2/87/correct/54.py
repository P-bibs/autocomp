# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150347/code_9.py
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
# CUDA kernel source – weight and input caching in shared memory
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

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
    const float sub2)
{
    const int out_h = in_h - k + 1;
    const int out_w = in_w - k + 1;

    const int oc = blockIdx.z;              // output channel for this block
    const int b  = blockIdx.y;              // batch index
    const int block_id_x = blockIdx.x % ((out_w + blockDim.x - 1) / blockDim.x);
    const int block_id_y = blockIdx.x / ((out_w + blockDim.x - 1) / blockDim.x);

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int TILE_W = blockDim.x;
    const int TILE_H = blockDim.y;
    const int tile_w = TILE_W + k - 1;
    const int tile_h = TILE_H + k - 1;

    // Optional padding to avoid bank conflicts
    const int tile_w_padded = ((tile_w + 31) / 32) * 32;

    extern __shared__ float shmem[];
    float* weight_cache = shmem;
    float* input_tile   = shmem + k * k * in_c;

    // ---- Load weight tile for this output channel into shared memory ----
    const int weight_offset = oc * (k * k * in_c);
    for (int idx = tid; idx < k * k * in_c; idx += blockDim.x * blockDim.y) {
        weight_cache[idx] = __ldg(&weight[weight_offset + idx]);
    }
    __syncthreads();

    // ---- Compute top-left corner of output tile for this block ----
    const int out_x0 = block_id_x * TILE_W;
    const int out_y0 = block_id_y * TILE_H;

    // ---- Load input tile into shared memory ----
    const float* i_base = input + (b * in_c * in_h * in_w);

    for (int ic = 0; ic < in_c; ++ic) {
        for (int i = 0; i < tile_h; i += TILE_H) {
            for (int j = 0; j < tile_w; j += TILE_W) {
                const int sx = tx + j;
                const int sy = ty + i;

                const int global_x = out_x0 + sx;
                const int global_y = out_y0 + sy;

                float val = 0.0f;
                if (global_x < in_w && global_y < in_h) {
                    val = __ldg(&i_base[ic * in_h * in_w + global_y * in_w + global_x]);
                }
                if (sx < tile_w && sy < tile_h) {
                    input_tile[(ic * tile_h + sy) * tile_w_padded + sx] = val;
                }
            }
        }
    }
    __syncthreads();

    // ---- Compute output position for this thread ----
    const int ow = out_x0 + tx;
    const int oh = out_y0 + ty;

    if (ow < out_w && oh < out_h) {
        float acc = __ldg(&bias[oc]);

        // Convolution using cached input and weight tiles
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                const int in_sx = tx + j;
                const int in_sy = ty + i;

                for (int ic = 0; ic < in_c; ++ic) {
                    const float inp = input_tile[(ic * tile_h + in_sy) * tile_w_padded + in_sx];
                    const float w = weight_cache[(i * k + j) * in_c + ic];
                    acc += inp * w;
                }
            }
        }

        // Mish activation after subtraction
        const float val = acc - sub1 - sub2;
        const float mish = val * tanhf(logf(1.0f + expf(val)));

        // Store result
        const int out_idx = ((b * out_c + oc) * out_h + oh) * out_w + ow;
        output[out_idx] = mish;
    }
}

// Host wrapper that launches the kernel with dynamic shared memory
void fused_conv_mish(torch::Tensor input,
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

    const int TILE_W = 16;
    const int TILE_H = 16;
    const int tile_w = TILE_W + k - 1;
    const int tile_h = TILE_H + k - 1;
    const int tile_w_padded = ((tile_w + 31) / 32) * 32;

    dim3 block(TILE_W, TILE_H);
    const int grid_x = (out_w + TILE_W - 1) / TILE_W;
    const int grid_y = (out_h + TILE_H - 1) / TILE_H;
    const int grid_out_xy = grid_x * grid_y;
    dim3 grid(grid_out_xy, batch, out_c);

    const size_t weight_mem = k * k * in_c * sizeof(float);
    const size_t input_tile_mem = in_c * tile_h * tile_w_padded * sizeof(float);
    const size_t shared_mem = weight_mem + input_tile_mem;

    fused_conv_mish_kernel<<<grid, block, shared_mem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_c, in_h, in_w,
        out_c, k, sub1, sub2);
}
"""

# ----------------------------------------------------------------------
# C++ binding (PyBind11)
# ----------------------------------------------------------------------
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
          "Fused convolution followed by Mish activation");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_conv_mish',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional model – entry point used for evaluation
# ----------------------------------------------------------------------
def functional_model(x: torch.Tensor,
                     *,
                     conv_weight: torch.Tensor,
                     conv_bias: torch.Tensor,
                     conv_stride: int = 1,
                     conv_padding: int = 0,
                     conv_dilation: int = 1,
                     conv_groups: int = 1,
                     subtract_value_1: float,
                     subtract_value_2: float) -> torch.Tensor:
    """
    Performs a fused convolution + Mish activation.
    Weight is given in PyTorch's default layout [out_c, in_c, kH, kW];
    we reorder it to [out_c, kH, kW, in_c] for a simple flat pointer.
    """
    # Reorder weight to [out_c, k, k, in_c] (flattened later)
    w_reordered = conv_weight.permute(0, 2, 3, 1).contiguous()

    batch, _, h, w = x.shape
    k = conv_weight.shape[2]
    out_h = h - k + 1
    out_w = w - k + 1

    # Allocate output tensor
    out = torch.empty((batch, conv_weight.size(0), out_h, out_w),
                      dtype=x.dtype, device=x.device)

    # Launch the custom fused kernel
    fused_ext.fused_conv(x, w_reordered, conv_bias, out,
                         subtract_value_1, subtract_value_2)

    return out
