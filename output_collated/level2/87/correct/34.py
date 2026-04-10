# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_144040/code_31.py
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
# CUDA source – Tiled shared-memory convolution for 3x3 kernel focus
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_H 4
#define TILE_W 4

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int in_c, int in_h, int in_w,
    int out_c, int k, int out_h, int out_w,
    float sub1, float sub2) {

    extern __shared__ float sdata[];

    // Tile configuration
    int region_h = TILE_H + k - 1;
    int region_w = TILE_W + k - 1;
    
    float* input_tile = sdata;
    float* weight_tile = sdata + (in_c * region_h * region_w);

    // Grid coordinates
    int tile_h_cnt = (out_h + TILE_H - 1) / TILE_H;
    int tile_w_cnt = (out_w + TILE_W - 1) / TILE_W;

    int idx = blockIdx.x;
    int b = idx / (out_c * tile_h_cnt * tile_w_cnt);
    idx %= (out_c * tile_h_cnt * tile_w_cnt);
    int oc = idx / (tile_h_cnt * tile_w_cnt);
    idx %= (tile_h_cnt * tile_w_cnt);
    int th = idx / tile_w_cnt;
    int tw = idx % tile_w_cnt;

    int oh_base = th * TILE_H;
    int ow_base = tw * TILE_W;

    // Load Weights into Shared Memory (Global: [out_c, in_c, k, k])
    int weight_vol = in_c * k * k;
    for (int i = threadIdx.x; i < weight_vol; i += blockDim.x) {
        weight_tile[i] = weight[oc * weight_vol + i];
    }

    // Load Input into Shared Memory
    int in_vol = in_c * region_h * region_w;
    int batch_offset = b * in_c * in_h * in_w;
    for (int i = threadIdx.x; i < in_vol; i += blockDim.x) {
        int ic = i / (region_h * region_w);
        int rem = i % (region_h * region_w);
        int rh = rem / region_w;
        int rw = rem % region_w;
        int ih = oh_base + rh;
        int iw = ow_base + rw;
        
        if (ih < in_h && iw < in_w)
            input_tile[i] = input[batch_offset + (ic * in_h + ih) * in_w + iw];
        else
            input_tile[i] = 0.0f;
    }

    __syncthreads();

    // Compute Output
    int ty = threadIdx.x / TILE_W;
    int tx = threadIdx.x % TILE_W;
    int oh = oh_base + ty;
    int ow = ow_base + tx;

    if (ty < TILE_H && tx < TILE_W && oh < out_h && ow < out_w) {
        float acc = bias[oc];
        for (int ic = 0; ic < in_c; ++ic) {
            for (int i = 0; i < k; ++i) {
                for (int j = 0; j < k; ++j) {
                    acc += input_tile[(ic * region_h + ty + i) * region_w + (tx + j)] * 
                           weight_tile[(ic * k + i) * k + j];
                }
            }
        }
        float val = acc - sub1 - sub2;
        output[((b * out_c + oc) * out_h + oh) * out_w + ow] = val * tanhf(logf(1.0f + expf(val)));
    }
}

void launch_fused_conv_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                            torch::Tensor output, float sub1, float sub2) {
    int b = input.size(0), ic = input.size(1), ih = input.size(2), iw = input.size(3);
    int oc = weight.size(0), k = weight.size(2);
    int oh = ih - k + 1, ow = iw - k + 1;
    
    int tile_h_cnt = (oh + TILE_H - 1) / TILE_H;
    int tile_w_cnt = (ow + TILE_W - 1) / TILE_W;
    int blocks = b * oc * tile_h_cnt * tile_w_cnt;
    size_t shared_size = (ic * (TILE_H + k - 1) * (TILE_W + k - 1) + ic * k * k) * sizeof(float);
    
    fused_conv_mish_kernel<<<blocks, 16, shared_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), b, ic, ih, iw, oc, k, oh, ow, sub1, sub2);
}
"""

cpp_source = r"""
void launch_fused_conv_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                            torch::Tensor output, float sub1, float sub2);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_mish", &launch_fused_conv_mish);
}
"""

module = load_inline(name='fused_conv_mish_ext', cpp_sources=cpp_source, cuda_sources=cuda_source, 
                     extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=0, 
                     conv_dilation=1, conv_groups=1, subtract_value_1, subtract_value_2):
    x, w, b = x.float().contiguous(), conv_weight.float().contiguous(), conv_bias.float().contiguous()
    out = torch.empty((x.shape[0], w.shape[0], x.shape[2]-w.shape[2]+1, x.shape[3]-w.shape[3]+1),
                      device=x.device, dtype=x.dtype)
    module.fused_conv_mish(x, w, b, out, subtract_value_1, subtract_value_2)
    return out
