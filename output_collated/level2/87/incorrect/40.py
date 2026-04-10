# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_145317/code_5.py
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
# Optimized CUDA kernel: Input tiling + shared memory + coalesced access
# ----------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_DIM 16
#define HALO (TILE_DIM + 2)

__global__ void fused_conv_mish_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch, const int in_c, const int in_h, const int in_w,
    const int out_c, const int k, const int out_h, const int out_w,
    const float sub1, const float sub2) {

    extern __shared__ float s_input[];

    int oc = blockIdx.z;
    int b = blockIdx.y;
    int tile_id = blockIdx.x;

    // Each block processes a TILE_DIM x TILE_DIM output tile
    int tile_start_h = (tile_id / ((out_w + TILE_DIM - 1) / TILE_DIM)) * TILE_DIM;
    int tile_start_w = (tile_id % ((out_w + TILE_DIM - 1) / TILE_DIM)) * TILE_DIM;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Shared memory layout: [in_c][HALO][HALO]
    float* s_input_ch = &s_input[ty * HALO + tx];

    // Process all input channels for this tile
    for (int ic = 0; ic < in_c; ++ic) {
        // Load input tile with halo into shared memory cooperatively
        for (int i = ty; i < HALO; i += TILE_DIM) {
            for (int j = tx; j < HALO; j += TILE_DIM) {
                int ih = tile_start_h + i;
                int iw = tile_start_w + j;
                float val = 0.0f;
                if (ih < in_h && iw < in_w) {
                    val = input[((b * in_c + ic) * in_h + ih) * in_w + iw];
                }
                s_input_ch[i * HALO + j] = val;
            }
        }
        __syncthreads();

        // Load weight for this output channel and input channel
        float w_val = weight[(oc * in_c + ic) * k * k + ty * k + tx];

        // Compute convolution for the tile
        for (int oh = ty; oh < TILE_DIM && (tile_start_h + oh) < out_h; oh += TILE_DIM) {
            for (int ow = tx; ow < TILE_DIM && (tile_start_w + ow) < out_w; ow += TILE_DIM) {
                float acc = (ic == 0) ? bias[oc] : 0.0f;

                // Perform convolution at this point
                for (int ki = 0; ki < k; ++ki) {
                    for (int kj = 0; kj < k; ++kj) {
                        float in_val = s_input_ch[(oh + ki) * HALO + (ow + kj)];
                        acc += in_val * w_val;
                    }
                }

                // Mish activation function applied only once per output element
                if (ic == in_c - 1) {
                    float val = acc - sub1 - sub2;
                    float softplus_val = logf(1.0f + expf(val));
                    output[((b * out_c + oc) * out_h + (tile_start_h + oh)) * out_w + (tile_start_w + ow)] = val * tanhf(softplus_val);
                }
            }
        }
        __syncthreads();
    }
}

void fused_conv_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                     torch::Tensor output, float sub1, float sub2) {
    const int batch = input.size(0);
    const int in_c = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_c = weight.size(0);
    const int k = weight.size(2);
    const int out_h = in_h - k + 1;
    const int out_w = in_w - k + 1;

    // Grid dimensions
    int tiles_x = (out_w + TILE_DIM - 1) / TILE_DIM;
    int tiles_y = (out_h + TILE_DIM - 1) / TILE_DIM;
    int num_tiles = tiles_x * tiles_y;

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(num_tiles, batch, out_c);

    // Shared memory size: in_c * HALO * HALO floats per block
    size_t shared_size = in_c * HALO * HALO * sizeof(float);

    fused_conv_mish_tiled_kernel<<<grid, block, shared_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch, in_c, in_h, in_w, out_c, k,
        out_h, out_w, sub1, sub2);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_conv_mish(torch::Tensor i, torch::Tensor w, torch::Tensor b, torch::Tensor o, float s1, float s2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_mish", &fused_conv_mish, "Fused Convolution, Subtraction, and Mish");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=0,
                     conv_dilation=1, conv_groups=1, subtract_value_1, subtract_value_2):
    batch, _, h, w = x.shape
    k = conv_weight.shape[2]
    out_h, out_w = h - k + 1, w - k + 1
    out = torch.empty((batch, conv_weight.size(0), out_h, out_w), device=x.device)
    fused_ext.fused_conv_mish(x, conv_weight, conv_bias, out, subtract_value_1, subtract_value_2)
    return out
