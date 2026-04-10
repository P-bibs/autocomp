# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150347/code_22.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Optimized implementation using cache-friendly spatial unrolling
__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int B, const int C_in, const int H, const int W,
    const int C_out, const int k, const float sub1, const float sub2,
    const int H_out, const int W_out) 
{
    // Block mapping: Z for batch, Y for C_out
    const int b = blockIdx.z;
    const int oc = blockIdx.y;
    
    // Thread mapping: X/Y for spatial output
    const int out_y = blockIdx.x * blockDim.y + threadIdx.y;
    const int out_x = threadIdx.x; 

    if (out_x >= W_out || out_y >= H_out) return;

    float acc = bias[oc];
    
    // Weight pointer for this output channel
    const float* w_ptr = weight + (oc * C_in * k * k);
    
    // Optimized loop: ic is the inner loop to maintain contiguous access
    #pragma unroll
    for (int ic = 0; ic < C_in; ++ic) {
        const float* in_ptr = input + (b * C_in * H * W) + (ic * H * W);
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                acc += in_ptr[(out_y + i) * W + (out_x + j)] * w_ptr[(ic * k * k) + (i * k + j)];
            }
        }
    }
    
    // Mish: x * tanh(softplus(x)) -> x * tanh(log(1 + exp(x)))
    float x = acc - sub1 - sub2;
    output[((b * C_out + oc) * H_out + out_y) * W_out + out_x] = x * tanhf(log1pf(expf(x)));
}

void fused_conv_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                     torch::Tensor output, float s1, float s2) {
    const int B = input.size(0);
    const int C_in = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int C_out = weight.size(0);
    const int k = weight.size(2);
    const int H_out = output.size(2);
    const int W_out = output.size(3);

    // Threads per block: width mapped to x, height mapped to y
    dim3 block(W_out, 1);
    dim3 grid(H_out, C_out, B);
    
    fused_conv_mish_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), B, C_in, H, W, C_out, k, s1, s2, H_out, W_out
    );
}
"""

cpp_source = r"""
void fused_conv_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                     torch::Tensor output, float s1, float s2);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &fused_conv_mish, "Fused Conv Mish");
}
"""

fused_ext = load_inline(
    name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=0, 
                     conv_dilation=1, conv_groups=1, subtract_value_1, subtract_value_2):
    batch, _, h, w = x.shape
    k = conv_weight.shape[2]
    out = torch.empty((batch, conv_weight.size(0), h - k + 1, w - k + 1), device=x.device, dtype=x.dtype)
    
    # Kernel assumes [B, C_in, H, W] and weights [C_out, C_in, k, k]
    fused_ext.fused_conv(
        x.contiguous(), 
        conv_weight.contiguous(), 
        conv_bias.contiguous(), 
        out, 
        float(subtract_value_1), 
        float(subtract_value_2)
    )
    return out
