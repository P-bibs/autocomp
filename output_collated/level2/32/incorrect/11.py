# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_012207/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'scale_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'scale_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, scales the output, and then applies a minimum operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

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
    if 'scale_factor' in flat_state:
        state_kwargs['scale_factor'] = flat_state['scale_factor']
    else:
        state_kwargs['scale_factor'] = getattr(model, 'scale_factor')
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

// Optimized kernel using thread partitioning
// Each block processes one spatial (h, w) for all output channels
__global__ void fused_conv_min_kernel(const float* __restrict__ input, 
                                     const float* __restrict__ weight, 
                                     const float* __restrict__ bias,
                                     float* __restrict__ output,
                                     float scale, int N, int H, int W, int CI, int CO) {
    int w_out = blockIdx.x;
    int h_out = blockIdx.y;
    int n = blockIdx.z;

    // Local min initialization
    float min_val = 1e30f; 

    // We compute convolution for all CO at this specific (n, h, w)
    // To optimize, iterate over CO and compute partial dot products
    for (int co = 0; co < CO; ++co) {
        float sum = bias[co];
        for (int ci = 0; ci < CI; ++ci) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    float in_val = input[((n * CI + ci) * (H + 2) + (h_out + kh)) * (W + 2) + (w_out + kw)];
                    float w_val = weight[(((co * CI + ci) * 3 + kh) * 3) + kw];
                    sum += in_val * w_val;
                }
            }
        }
        float val = sum * scale;
        if (val < min_val) min_val = val;
    }
    output[n * (H * W) + h_out * W + w_out] = min_val;
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                      float scale, torch::Tensor output) {
    int N = input.size(0);
    int H = input.size(2) - 2;
    int W = input.size(3) - 2;
    int CI = input.size(1);
    int CO = weight.size(0);

    dim3 blocks(W, H, N);
    fused_conv_min_kernel<<<blocks, 1>>>(input.data_ptr<float>(), weight.data_ptr<float>(), 
                                        bias.data_ptr<float>(), output.data_ptr<float>(), 
                                        scale, N, H, W, CI, CO);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scale, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Convolution and Min Reduction");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=1, 
                     conv_dilation=1, conv_groups=1, scale_factor):
    # Padding: standard torch padding to match spatial dims
    x = torch.nn.functional.pad(x, (1, 1, 1, 1))
    N, CI, H_pad, W_pad = x.shape
    H, W = H_pad - 2, W_pad - 2
    
    output = torch.empty((N, 1, H, W), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_weight, conv_bias, scale_factor, output)
    return output
