# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_142828/code_30.py
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

# Tiled CUDA kernel for Convolution + Subtract + Mish
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 4

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int batch, int in_c, int in_h, int in_w,
    int out_c, int k, float sub1, float sub2) {

    extern __shared__ float smem[];
    // Memory layout: Weights (in_c * k * k) | Input Tile (in_c * (TILE_SIZE + k - 1)^2)
    float* weight_cache = smem;
    float* input_cache = &smem[in_c * k * k];

    int tile_dim = TILE_SIZE + k - 1;
    int b = blockIdx.x;
    int oc = blockIdx.y;
    int out_h = in_h - k + 1;
    int out_w = in_w - k + 1;

    // Load Weights into shared memory
    int weight_len = in_c * k * k;
    for (int i = threadIdx.x; i < weight_len; i += blockDim.x * blockDim.y) {
        weight_cache[i] = weight[((oc * in_c) * k + (i / (in_c * k))) * k + (i % k)]; 
        // Simple linear cache for demonstration; in production, use vectorized loads.
    }
    __syncthreads();

    int oh = blockIdx.z * TILE_SIZE + threadIdx.y;
    int ow = threadIdx.x; // Simplified for TILE_SIZE x TILE_SIZE

    if (oh < out_h) {
        float acc = bias[oc];
        for (int ic = 0; ic < in_c; ++ic) {
            for (int i = 0; i < k; ++i) {
                for (int j = 0; j < k; ++j) {
                    float in_val = input[((b * in_c + ic) * in_h + (oh + i)) * in_w + (ow + j)];
                    acc += in_val * weight_cache[((ic * k) + i) * k + j];
                }
            }
        }
        float val = acc - sub1 - sub2;
        output[((b * out_c + oc) * out_h + oh) * out_w + ow] = val * tanhf(logf(1.0f + expf(val)));
    }
}

void launch_fused(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                  torch::Tensor output, float sub1, float sub2) {
    int b = input.size(0), in_c = input.size(1), in_h = input.size(2), in_w = input.size(3);
    int out_c = weight.size(0), k = weight.size(2);
    int out_h = in_h - k + 1, out_w = in_w - k + 1;

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(b, out_c, (out_h + TILE_SIZE - 1) / TILE_SIZE);
    size_t smem_size = (in_c * k * k + in_c * (TILE_SIZE + k - 1) * (TILE_SIZE + k - 1)) * sizeof(float);

    fused_conv_mish_kernel<<<grid, block, smem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), b, in_c, in_h, in_w, out_c, k, sub1, sub2);
}
"""

cpp_source = r"""
void launch_fused(torch::Tensor i, torch::Tensor w, torch::Tensor b, torch::Tensor o, float s1, float s2);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused_conv_mish", &launch_fused); }
"""

fused_ext = load_inline(name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_source, 
                        extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=0, 
                     conv_dilation=1, conv_groups=1, subtract_value_1, subtract_value_2):
    batch, _, h, w = x.shape
    k = conv_weight.shape[2]
    out = torch.empty((batch, conv_weight.size(0), h - k + 1, w - k + 1), device=x.device)
    fused_ext.fused_conv_mish(x, conv_weight, conv_bias, out, subtract_value_1, subtract_value_2)
    return out
