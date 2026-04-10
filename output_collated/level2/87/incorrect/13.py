# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141921/code_2.py
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

#define TILE_SIZE 16
#define K_SIZE 3

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int batch, int in_c, int in_h, int in_w,
    int out_c, int k, float sub1, float sub2) {
    
    extern __shared__ float shared_input[];
    
    int out_h_val = in_h - k + 1;
    int out_w_val = in_w - k + 1;
    
    // Calculate thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int oc = blockIdx.z;
    int b = blockIdx.y;
    int ow = blockIdx.x * blockDim.x + tx;
    int oh = ty;
    
    // Load input data into shared memory in a coalesced manner
    // Each thread loads multiple elements if needed
    for (int ic = 0; ic < in_c; ++ic) {
        for (int i = 0; i < TILE_SIZE + K_SIZE - 1; i += blockDim.y) {
            int ih = oh + i;
            if (ih < TILE_SIZE + K_SIZE - 1 && ow < TILE_SIZE + K_SIZE - 1) {
                int global_ih = blockIdx.x * TILE_SIZE + ih;
                int global_iw = blockIdx.x * TILE_SIZE + ow;
                
                if (global_ih < in_h && global_iw < in_w) {
                    shared_input[(ic * (TILE_SIZE + K_SIZE - 1) + ih) * (TILE_SIZE + K_SIZE - 1) + ow] = 
                        input[((b * in_c + ic) * in_h + global_ih) * in_w + global_iw];
                } else {
                    shared_input[(ic * (TILE_SIZE + K_SIZE - 1) + ih) * (TILE_SIZE + K_SIZE - 1) + ow] = 0.0f;
                }
            }
        }
    }
    __syncthreads();
    
    // Perform convolution with data from shared memory
    if (ow < out_w_val) {
        for (int oh_local = 0; oh_local < TILE_SIZE; ++oh_local) {
            int global_oh = blockIdx.x * TILE_SIZE + oh_local;
            if (global_oh < out_h_val) {
                float acc = bias[oc];
                
                // Convolution computation
                for (int ic = 0; ic < in_c; ++ic) {
                    for (int i = 0; i < k; ++i) {
                        for (int j = 0; j < k; ++j) {
                            float in_val = shared_input[(ic * (TILE_SIZE + K_SIZE - 1) + oh_local + i) * (TILE_SIZE + K_SIZE - 1) + (ow % (TILE_SIZE + K_SIZE - 1)) + j];
                            float w_val = weight[((oc * in_c + ic) * k + i) * k + j];
                            acc += in_val * w_val;
                        }
                    }
                }
                
                // Apply Sub + Mish
                float val = acc - sub1 - sub2;
                output[((b * out_c + oc) * out_h_val + global_oh) * out_w_val + ow] = val * tanhf(logf(1.0f + expf(val)));
            }
        }
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
    
    // Grid and block dimensions
    dim3 block(TILE_SIZE + K_SIZE - 1, TILE_SIZE);
    dim3 grid((out_w + block.x - 1) / block.x, batch, out_c);
    
    // Shared memory size calculation
    size_t shared_size = in_c * (TILE_SIZE + K_SIZE - 1) * (TILE_SIZE + K_SIZE - 1) * sizeof(float);
    
    fused_conv_mish_kernel<<<grid, block, shared_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch, in_c, in_h, in_w, out_c, k, sub1, sub2);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                     torch::Tensor output, float sub1, float sub2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_mish", &fused_conv_mish, "Fused Convolution, Subtraction, and Mish with Shared Memory Optimization");
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
    out_h = h - k + 1
    out_w = w - k + 1
    out_c = conv_weight.shape[0]
    out = torch.empty((batch, out_c, out_h, out_w), device=x.device)
    fused_ext.fused_conv_mish(x, conv_weight, conv_bias, out, subtract_value_1, subtract_value_2)
    return out
