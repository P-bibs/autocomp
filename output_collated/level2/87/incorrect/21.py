# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_142828/code_6.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_DIM 16
#define HALO_RADIUS 1 // For 3x3 kernel

__global__ void fused_conv_mish_kernel_tiled(
    const float* __restrict__ input, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int batch, int in_c, int in_h, int in_w,
    int out_c, int k, float sub1, float sub2) {

    // Shared memory for input tile with halo
    extern __shared__ float s_input[];
    
    int out_h_val = in_h - k + 1;
    int out_w_val = in_w - k + 1;
    
    // Block and thread indices
    int oc = blockIdx.z;
    int b = blockIdx.y;
    int block_oh = blockIdx.x / ((out_w_val + TILE_DIM - 1) / TILE_DIM);
    int block_ow = blockIdx.x % ((out_w_val + TILE_DIM - 1) / TILE_DIM);
    
    int tid_x = threadIdx.x % TILE_DIM;
    int tid_y = threadIdx.x / TILE_DIM;
    
    // Calculate global output position
    int oh = block_oh * TILE_DIM + tid_y;
    int ow = block_ow * TILE_DIM + tid_x;

    // Shared memory dimensions (includes halo)
    int s_h = TILE_DIM + 2 * HALO_RADIUS;
    int s_w = TILE_DIM + 2 * HALO_RADIUS;
    
    float acc = 0.0f;
    
    // Loop over input channels
    for (int ic = 0; ic < in_c; ++ic) {
        // Load input tile with halo into shared memory
        for (int iy = tid_y; iy < s_h; iy += TILE_DIM) {
            for (int ix = tid_x; ix < s_w; ix += TILE_DIM) {
                int in_y = block_oh * TILE_DIM + iy - HALO_RADIUS;
                int in_x = block_ow * TILE_DIM + ix - HALO_RADIUS;
                
                // Clamp to valid input range
                in_y = max(0, min(in_y, in_h - 1));
                in_x = max(0, min(in_x, in_w - 1));
                
                s_input[iy * s_w + ix] = input[((b * in_c + ic) * in_h + in_y) * in_w + in_x];
            }
        }
        __syncthreads();

        // Perform convolution if within output bounds
        if (oh < out_h_val && ow < out_w_val) {
            for (int i = 0; i < k; ++i) {
                for (int j = 0; j < k; ++j) {
                    float in_val = s_input[(tid_y + i) * s_w + (tid_x + j)];
                    float w_val = weight[((oc * in_c + ic) * k + i) * k + j];
                    acc += in_val * w_val;
                }
            }
        }
        __syncthreads();
    }
    
    // Apply bias, subtract, and Mish activation
    if (oh < out_h_val && ow < out_w_val) {
        float val = acc + bias[oc] - sub1 - sub2;
        // Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
        output[((b * out_c + oc) * out_h_val + oh) * out_w_val + ow] = val * tanhf(logf(1.0f + expf(val)));
    }
}

void fused_conv_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                     torch::Tensor output, float sub1, float sub2) {
    int batch = input.size(0);
    int in_c = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_c = weight.size(0);
    int k = weight.size(2);
    int out_h = in_h - k + 1;
    int out_w = in_w - k + 1;
    
    // For 3x3 kernel, we have a halo radius of 1
    int s_h = TILE_DIM + 2 * HALO_RADIUS;
    int s_w = TILE_DIM + 2 * HALO_RADIUS;
    int shared_mem_size = s_h * s_w * sizeof(float);
    
    // Grid dimensions
    int grid_x = ((out_h + TILE_DIM - 1) / TILE_DIM) * ((out_w + TILE_DIM - 1) / TILE_DIM);
    dim3 blocks(grid_x, batch, out_c);
    int threads = TILE_DIM * TILE_DIM;
    
    fused_conv_mish_kernel_tiled<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch, in_c, in_h, in_w, out_c, k, sub1, sub2);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_conv_mish(torch::Tensor i, torch::Tensor w, torch::Tensor b, torch::Tensor o, float s1, float s2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_mish", &fused_conv_mish, "Tiled Fused Convolution, Subtraction, and Mish");
}
"""

fused_ext = load_inline(
    name='fused_ext_tiled', 
    cpp_sources=cpp_source, 
    cuda_sources=cuda_source, 
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=0, 
                     conv_dilation=1, conv_groups=1, subtract_value_1, subtract_value_2):
    batch, _, h, w = x.shape
    k = conv_weight.shape[2]
    out = torch.empty((batch, conv_weight.size(0), h - k + 1, w - k + 1), device=x.device)
    fused_ext.fused_conv_mish(x, conv_weight, conv_bias, out, subtract_value_1, subtract_value_2)
    return out
