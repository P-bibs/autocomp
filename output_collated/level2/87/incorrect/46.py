# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150347/code_4.py
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

# CUDA kernel: Optimized version with tiled shared memory for inputs and weights
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_DIM 16
#define HALO (KSIZE-1)

template<int KSIZE>
__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int in_c, int in_h, int in_w,
    int out_h, int out_w, 
    float sub1, float sub2)
{
    // Shared memory for input tile with halo
    __shared__ float input_tile[TILE_DIM + HALO][TILE_DIM + HALO][32]; // Assumes in_c <= 32
    // Shared memory for weights
    __shared__ float weight_tile[KSIZE * KSIZE * 32]; 

    const int oc = blockIdx.y;
    const int b = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Load weights into shared memory cooperatively
    const int weight_vol = KSIZE * KSIZE * in_c;
    for (int i = ty * blockDim.x + tx; i < weight_vol; i += blockDim.x * blockDim.y) {
        weight_tile[i] = weight[oc * weight_vol + i];
    }
    __syncthreads();

    const int oh_base = blockIdx.x * TILE_DIM;
    const int ow_base = 0; // Process full width in tiled manner or use a 2D block scheme if needed

    // Load input tile with halo into shared memory
    // Each thread loads multiple elements if necessary
    const int ih_base = oh_base;
    const int iw_base = ow_base;
    
    // Clamp to avoid out-of-bounds reads
    for (int ic = ty; ic < in_c; ic += blockDim.y) {
        for (int ih = ih_base + tx; ih < min(ih_base + TILE_DIM + HALO, in_h); ih += blockDim.x) {
            for (int iw = iw_base; iw < min(iw_base + TILE_DIM + HALO, in_w); iw++) {
                if (ih < in_h && iw < in_w) {
                    input_tile[ih - ih_base][iw - iw_base][ic] = 
                        input[((b * in_c + ic) * in_h + ih) * in_w + iw];
                }
            }
        }
    }
    __syncthreads();

    // Compute output
    if (oh_base + ty < out_h && tx < out_w) {
        const int oh = oh_base + ty;
        const int ow = tx;
        float acc = bias[oc];

        // Convolve with loaded tile
        for (int kh = 0; kh < KSIZE; ++kh) {
            for (int kw = 0; kw < KSIZE; ++kw) {
                const int ih = oh + kh;
                const int iw = ow + kw;
                if (ih < in_h && iw < in_w) {
                    const int w_idx_base = (kh * KSIZE + kw) * in_c;
                    #pragma unroll 4
                    for (int ic = 0; ic < in_c; ++ic) {
                        acc += input_tile[ih - ih_base][iw - iw_base][ic] * weight_tile[w_idx_base + ic];
                    }
                }
            }
        }

        // Mish activation
        float val = acc - sub1 - sub2;
        float activated = val * tanhf(logf(1.0f + expf(val)));

        output[((b * gridDim.y + oc) * out_h + oh) * out_w + ow] = activated;
    }
}

void fused_conv_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                     torch::Tensor output, float sub1, float sub2) {
    const int batch = input.size(0);
    const int in_c = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_c = weight.size(0);
    const int k = weight.size(2); // Assuming square kernel [out_c, in_c, k, k]
    const int out_h = in_h - k + 1;
    const int out_w = in_w - k + 1;

    dim3 threads(TILE_DIM, TILE_DIM);
    const int blocks_x = (out_h + TILE_DIM - 1) / TILE_DIM;
    dim3 blocks(blocks_x, out_c, batch);

    // Launch templated kernel
    switch(k) {
        case 1: fused_conv_mish_kernel<1><<<blocks, threads>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                                                              output.data_ptr<float>(), in_c, in_h, in_w, out_h, out_w, sub1, sub2); break;
        case 3: fused_conv_mish_kernel<3><<<blocks, threads>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                                                              output.data_ptr<float>(), in_c, in_h, in_w, out_h, out_w, sub1, sub2); break;
        case 5: fused_conv_mish_kernel<5><<<blocks, threads>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                                                              output.data_ptr<float>(), in_c, in_h, in_w, out_h, out_w, sub1, sub2); break;
        default: AT_ERROR("Unsupported kernel size");
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_conv_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                     torch::Tensor output, float sub1, float sub2);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &fused_conv_mish, "Fused Convolution + Mish with Tiled Shared Memory");
}
"""

# Compile the extension at import time
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-DTILE_DIM=16'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=0, 
                     conv_dilation=1, conv_groups=1, subtract_value_1, subtract_value_2):
    """
    Functional model implementation using a custom CUDA fused kernel.
    
    Args:
        x: Input tensor of shape [N, C_in, H_in, W_in]
        conv_weight: Weight tensor of shape [C_out, C_in, K, K]
        conv_bias: Bias tensor of shape [C_out]
        subtract_value_1, subtract_value_2: Scalars subtracted before Mish activation
    
    Returns:
        Output tensor of shape [N, C_out, H_out, W_out]
    """
    assert conv_stride == 1 and conv_padding == 0 and conv_dilation == 1 and conv_groups == 1, \
        "Only default convolution parameters supported"
        
    w = conv_weight.contiguous()
    k = conv_weight.shape[2]
    batch, in_c, h, w_dim = x.shape
    out_h, out_w = h - k + 1, w_dim - k + 1
    out = torch.empty((batch, conv_weight.size(0), out_h, out_w), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_conv(x, w, conv_bias, out, float(subtract_value_1), float(subtract_value_2))
    return out
