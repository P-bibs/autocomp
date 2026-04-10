# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_145317/code_4.py
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

# Optimized CUDA kernel using NHWC layout for coalesced memory access
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE 16

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int in_c, int in_h, int in_w,
    int out_c, int k, int out_h, int out_w,
    float sub1, float sub2)
{
    // Each block processes a tile of the spatial output map for a specific (batch, output_channel)
    int oc = blockIdx.y;
    int b = blockIdx.z;
    int tile_h = blockIdx.x / ((out_w + TILE - 1) / TILE);
    int tile_w = blockIdx.x % ((out_w + TILE - 1) / TILE);

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int oh = tile_h * TILE + ty;
    int ow = tile_w * TILE + tx;

    if (oh < out_h && ow < out_w) {
        float acc = bias[oc];
        int weight_oc_offset = oc * k * k * in_c;

        // Convolution loop with NHWC input access for coalesced memory transactions
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                // Calculate base address in NHWC format: (batch, height, width, channel)
                int ih = oh + i;
                int iw = ow + j;
                
                // Input layout: [batch][height][width][channel]
                int input_base = ((b * in_h + ih) * in_w + iw) * in_c;
                int weight_base = (i * k + j) * in_c;
                
                // Vectorized loading or unroll can be applied here if needed
                for (int ic = 0; ic < in_c; ++ic) {
                    acc += input[input_base + ic] * weight[weight_oc_offset + weight_base + ic];
                }
            }
        }

        // Mish activation: f(x) = x * tanh(ln(1 + e^x))
        // Apply the subtraction values as part of the bias adjustment
        float val = acc - sub1 - sub2;
        float exp_val = expf(val);
        float tanh_val = tanhf(logf(1.0f + exp_val));
        output[((b * out_c + oc) * out_h + oh) * out_w + ow] = val * tanh_val;
    }
}

void fused_conv_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                     torch::Tensor output, float sub1, float sub2) {
    int batch = input.size(0);
    int in_h = input.size(1);
    int in_w = input.size(2);
    int in_c = input.size(3);
    int out_c = weight.size(0);
    int k = weight.size(2); // Assuming square kernel [out_c][k][k][in_c]
    int out_h = in_h - k + 1;
    int out_w = in_w - k + 1;

    dim3 threads(TILE, TILE);
    dim3 blocks(
        ((out_h + TILE - 1) / TILE) * ((out_w + TILE - 1) / TILE), 
        out_c, 
        batch
    );
    
    fused_conv_mish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch, in_c, in_h, in_w, out_c, k, out_h, out_w, sub1, sub2
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_conv_mish(torch::Tensor i, torch::Tensor w, torch::Tensor b, torch::Tensor o, float s1, float s2);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &fused_conv_mish, "Fused Convolution + Mish");
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

def functional_model(x, *, conv_weight, conv_bias, subtract_value_1=0.0, subtract_value_2=0.0):
    # Convert NCHW to NHWC format for coalesced memory access in kernel
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    w_nhwc = conv_weight.permute(0, 2, 3, 1).contiguous()
    
    batch = x.shape[0]
    in_h, in_w = x.shape[2], x.shape[3]
    out_c = conv_weight.shape[0]
    k = conv_weight.shape[2]
    out_h = in_h - k + 1
    out_w = in_w - k + 1
    
    # Allocate output tensor with correct dimensions
    out = torch.empty((batch, out_h, out_w, out_c), device=x.device, dtype=x.dtype)
    
    # Launch fused kernel
    fused_ext.fused_conv(x_nhwc, w_nhwc, conv_bias, out, subtract_value_1, subtract_value_2)
    
    # Convert back to NCHW format as expected by PyTorch convention
    return out.permute(0, 3, 1, 2).contiguous()
