# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_142828/code_12.py
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
# Optimized CUDA kernel: tiled convolution with shared-memory input cache
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_DIM 8
#define GROUP_SIZE 8

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch, const int in_c, const int in_h, const int in_w,
    const int out_c, const int k,
    const int out_h, const int out_w,
    const float sub1, const float sub2) {

    // Shared memory layout: weights + input patches
    extern __shared__ float shared_mem[];
    const int weight_size = in_c * k * k;
    float* s_weight = shared_mem;
    float* s_input = &shared_mem[weight_size];

    // Block indices
    const int tile_x = blockIdx.x;
    const int tile_y = blockIdx.y % ((out_h + TILE_DIM - 1) / TILE_DIM);
    const int oc = blockIdx.y / ((out_h + TILE_DIM - 1) / TILE_DIM);
    const int b = blockIdx.z;

    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int block_size = blockDim.x * blockDim.y;

    // Load weights for current output channel into shared memory
    for (int i = tid; i < weight_size; i += block_size) {
        s_weight[i] = weight[oc * weight_size + i];
    }
    __syncthreads();

    // Output position for this thread
    const int oh = tile_y * TILE_DIM + ty;
    const int ow = tile_x * TILE_DIM + tx;

    // Early exit if out of bounds
    if (oh >= out_h || ow >= out_w) return;

    // Top-left corner of input region (including padding)
    const int i_start = tile_y * TILE_DIM;
    const int j_start = tile_x * TILE_DIM;

    float acc = 0.0f;

    // Process input channels in groups
    for (int ic0 = 0; ic0 < in_c; ic0 += GROUP_SIZE) {
        const int gch = min(GROUP_SIZE, in_c - ic0);
        const int region_h = TILE_DIM + k - 1;
        const int region_w = TILE_DIM + k - 1;
        const int region_size = gch * region_h * region_w;

        // Load input patch for this group into shared memory
        for (int load_idx = tid; load_idx < region_size; load_idx += block_size) {
            const int c = load_idx / (region_h * region_w);
            const int rem = load_idx % (region_h * region_w);
            const int r = rem / region_w;
            const int cc = rem % region_w;

            const int ic = ic0 + c;
            const int i_global = i_start + r;
            const int j_global = j_start + cc;

            if (ic < in_c && i_global < in_h && j_global < in_w) {
                s_input[c * region_h * region_w + r * region_w + cc] =
                    input[((b * in_c + ic) * in_h + i_global) * in_w + j_global];
            } else {
                s_input[c * region_h * region_w + r * region_w + cc] = 0.0f;
            }
        }
        __syncthreads();

        // Compute convolution for this group
        for (int ic = 0; ic < gch; ++ic) {
            const int c_idx = ic;
            for (int ki = 0; ki < k; ++ki) {
                for (int kj = 0; kj < k; ++kj) {
                    const float in_val = s_input[c_idx * region_h * region_w +
                                                (ty + ki) * region_w +
                                                (tx + kj)];
                    const int w_idx = ((ic0 + ic) * k + ki) * k + kj;
                    const float w_val = s_weight[w_idx];
                    acc += in_val * w_val;
                }
            }
        }
        __syncthreads(); // Synchronize before overwriting s_input
    }

    // Add bias and apply fused Mish activation
    acc += bias[oc];
    const float val = acc - sub1 - sub2;
    const float mish = val * tanhf(logf(1.0f + expf(val)));

    // Write output
    output[((b * out_c + oc) * out_h + oh) * out_w + ow] = mish;
}

void fused_conv_mish(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, float sub1, float sub2) {
    
    const int batch = input.size(0);
    const int in_c = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_c = weight.size(0);
    const int k = weight.size(2);
    const int out_h = in_h - k + 1;
    const int out_w = in_w - k + 1;

    // Tile dimensions
    const int tile_h = TILE_DIM;
    const int tile_w = TILE_DIM;
    
    // Grid dimensions
    const int tile_x_cnt = (out_w + tile_w - 1) / tile_w;
    const int tile_y_cnt = (out_h + tile_h - 1) / tile_h;
    
    dim3 block(tile_w, tile_h);
    dim3 grid(tile_x_cnt, out_c * tile_y_cnt, batch);

    // Shared memory size
    const int weight_size = in_c * k * k;
    const int region_h = tile_h + k - 1;
    const int region_w = tile_w + k - 1;
    const size_t shared_size = (weight_size + GROUP_SIZE * region_h * region_w) * sizeof(float);

    fused_conv_mish_kernel<<<grid, block, shared_size>>>(
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

void fused_conv_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                     torch::Tensor output, float sub1, float sub2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_mish", &fused_conv_mish, "Fused Convolution + Mish with Tiling");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias,
                     conv_stride=1, conv_padding=0,
                     conv_dilation=1, conv_groups=1,
                     subtract_value_1, subtract_value_2):
    """
    Fused convolution (stride=1, no padding, no dilation, groups=1) followed by
    elementwise subtraction of two scalars and Mish activation.
    """
    batch, in_c, in_h, in_w = x.shape
    k = conv_weight.shape[2]  # square kernel assumed
    out_h = in_h - k + 1
    out_w = in_w - k + 1

    # Allocate output
    out = torch.empty((batch, conv_weight.size(0), out_h, out_w),
                      device=x.device, dtype=x.dtype)

    # Call the custom CUDA kernel
    fused_ext.fused_conv_mish(x, conv_weight, conv_bias, out,
                              subtract_value_1, subtract_value_2)

    return out
