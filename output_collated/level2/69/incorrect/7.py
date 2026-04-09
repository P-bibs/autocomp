# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_050613/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies HardSwish, and then ReLU.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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

# The CUDA kernel performs a direct convolution and fuses the activations 
# (hardswish then relu) into a single write-back to global memory.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_act_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int n, int in_c, int out_c, int h, int w, int k) {
    
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z; // output channel

    int out_h = h - k + 1;
    int out_w = w - k + 1;

    if (ox >= out_w || oy >= out_h) return;

    float val = bias[oc];
    for (int ic = 0; ic < in_c; ++ic) {
        for (int ky = 0; ky < k; ++ky) {
            for (int kx = 0; kx < k; ++kx) {
                float in_val = input[((0 * in_c + ic) * h + (oy + ky)) * w + (ox + kx)];
                float w_val = weight[(((oc * in_c + ic) * k + ky) * k + kx)];
                val += in_val * w_val;
            }
        }
    }

    // Fused activation: hardswish(x) = x * min(relu6(x + 3), 6) / 6
    float hswish = val * fminf(fmaxf(val + 3.0f, 0.0f), 6.0f) / 6.0f;
    // Followed by standard relu
    output[((0 * out_c + oc) * out_h + oy) * out_w + ox] = (hswish > 0.0f) ? hswish : 0.0f;
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    int b = input.size(0);
    int in_c = input.size(1);
    int h = input.size(2);
    int w = input.size(3);
    int out_c = weight.size(0);
    int k = weight.size(2);
    int out_h = h - k + 1;
    int out_w = w - k + 1;

    dim3 threads(16, 16);
    dim3 blocks((out_w + 15) / 16, (out_h + 15) / 16, out_c);
    
    fused_conv_act_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), b, in_c, out_c, h, w, k);
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Convolution Activation Kernel");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1):
    # This implementation assumes the input layout provided in the prompt
    out_h = x.shape[2] - conv_weight.shape[2] + 1
    out_w = x.shape[3] - conv_weight.shape[3] + 1
    output = torch.empty((x.shape[0], conv_weight.shape[0], out_h, out_w), device=x.device)
    fused_ext.fused_op(x, conv_weight, conv_bias, output)
    return output
