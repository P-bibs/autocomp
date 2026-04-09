# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_051208/code_5.py
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

# The custom CUDA kernel implements a direct convolution (sliding window) followed by 
# HardSwish and ReLU fusion. This avoids global memory writes for intermediate 
# convolution results, performing activations directly in registers.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ __forceinline__ float hardswish(float x) {
    // x * min(max(x + 3, 0), 6) / 6
    float val = x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) * 0.166666667f;
    return (val > 0.0f) ? val : 0.0f; // Fused ReLU
}

__global__ void fused_conv_act_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias,
    float* __restrict__ output, 
    int N, int C, int H, int W, int OC, int K) {
    
    int oc = blockIdx.x; 
    int n = blockIdx.y;
    int hw = blockIdx.z * blockDim.x + threadIdx.x;
    
    int OH = H - K + 1;
    int OW = W - K + 1;
    
    if (hw >= OH * OW) return;

    int out_h = hw / OW;
    int out_w = hw % OW;
    
    float acc = bias[oc];
    // Direct convolution loops
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int ih = out_h + kh;
                int iw = out_w + kw;
                acc += input[((n * C + c) * H + ih) * W + iw] * 
                       weight[((oc * C + c) * K + kh) * K + kw];
            }
        }
    }
    
    output[((n * OC + oc) * OH + out_h) * OW + out_w] = hardswish(acc);
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int OC = weight.size(0);
    const int K = weight.size(2);
    
    int OH = H - K + 1;
    int OW = W - K + 1;
    
    dim3 threads(128);
    dim3 blocks(OC, N, (OH * OW + threads.x - 1) / threads.x);
    
    fused_conv_act_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C, H, W, OC, K);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv2d + HardSwish + ReLU forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    # Only supports the specified configuration (stride 1, padding 0, dilation 1, groups 1)
    # as per the requirement for a high-performance fused implementation without built-ins.
    batch_size, _, height, width = x.shape
    out_channels, _, k, _ = conv_weight.shape
    out_h, out_w = height - k + 1, width - k + 1
    
    out = torch.empty((batch_size, out_channels, out_h, out_w), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(x, conv_weight, conv_bias, out)
    return out
