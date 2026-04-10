# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141921/code_9.py
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

# Optimized CUDA Kernel: Manual implementation of 2D Convolution + Bias + Subtractions + Mish
# Note: Manually writing convolution kernels is for illustrative purposes of fusion.
# For production, one would typically use cuDNN/cutlass wrappers, but this fulfills
# the requirement to replace built-in convolution functions with a custom kernel.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int Ci, int Co, int H, int W, int K, int S, int P,
    float sub1, float sub2) {
    
    // Grid maps to (B, Co, Ho, Wo)
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int co = blockIdx.z;

    int Ho = (H + 2 * P - K) / S + 1;
    int Wo = (W + 2 * P - K) / S + 1;

    if (out_w >= Wo || out_h >= Ho || co >= Co) return;

    for (int b = 0; b < B; ++b) {
        float val = 0.0f;
        int in_h_origin = out_h * S - P;
        int in_w_origin = out_w * S - P;

        for (int ci = 0; ci < Ci; ++ci) {
            for (int ky = 0; ky < K; ++ky) {
                int ih = in_h_origin + ky;
                if (ih < 0 || ih >= H) continue;
                for (int kx = 0; kx < K; ++kx) {
                    int iw = in_w_origin + kx;
                    if (iw < 0 || iw >= W) continue;
                    
                    float in_val = input[((b * Ci + ci) * H + ih) * W + iw];
                    float w_val = weight[((co * Ci + ci) * K + ky) * K + kx];
                    val += in_val * w_val;
                }
            }
        }

        val += bias[co];
        val = val - sub1 - sub2;
        
        // Mish: x * tanh(ln(1 + exp(x)))
        float m = val * tanhf(logf(1.0f + expf(val)));
        
        output[((b * Co + co) * Ho + out_h) * Wo + out_w] = m;
    }
}

void launch_fused_conv(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int S, int P, float sub1, float sub2) {
    
    int B = input.size(0);
    int Ci = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int Co = weight.size(0);
    int K = weight.size(2);
    int Ho = output.size(2);
    int Wo = output.size(3);

    dim3 block(16, 16);
    dim3 grid((Wo + 15) / 16, (Ho + 15) / 16, Co);

    fused_conv_mish_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), B, Ci, Co, H, W, K, S, P, sub1, sub2);
}
"""

cpp_source = r"""
void launch_fused_conv(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int S, int P, float sub1, float sub2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused_conv, "Fused Conv/Sub/Mish");
}
"""

fused_ext = load_inline(
    name='fused_op_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, 
                     conv_groups, subtract_value_1, subtract_value_2):
    # This implementation assumes standard parameters as per original script
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_weight.shape[0]
    kernel_size = conv_weight.shape[2]
    
    ho = (height + 2 * conv_padding - kernel_size) // conv_stride + 1
    wo = (width + 2 * conv_padding - kernel_size) // conv_stride + 1
    
    output = torch.empty((batch_size, out_channels, ho, wo), device='cuda', dtype=torch.float32)
    
    fused_ext.fused_op(
        x.contiguous(), conv_weight.contiguous(), conv_bias.contiguous(), output,
        conv_stride, conv_padding, float(subtract_value_1), float(subtract_value_2)
    )
    
    return output
