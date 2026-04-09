# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_082041/code_10.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x, const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ output, int N, int C_in, int C_out, int H, int W, int K,
    int stride, int padding, int OH, int OW) {

    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;

    if (oh >= OH || ow >= OW) return;

    float min_val = 1e38f;
    
    // Iterate over channels and kernel to compute each output channel's value
    for (int co = 0; co < C_out; ++co) {
        float sum = bias[co];
        for (int ci = 0; ci < C_in; ++ci) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int ih = oh * stride + kh - padding;
                    int iw = ow * stride + kw - padding;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        float val = x[((n * C_in + ci) * H + ih) * W + iw];
                        float w = weight[(((co * C_in + ci) * K + kh) * K + kw)];
                        sum += val * w;
                    }
                }
            }
        }
        if (co == 0) min_val = sum;
        else if (sum < min_val) min_val = sum;
    }
    
    // Double tanh activation
    float res = tanhf(tanhf(min_val));
    output[((n * OH + oh) * OW + ow)] = res;
}

void fused_conv_min_tanh(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
                         torch::Tensor output, int stride, int padding) {
    int N = x.size(0); int C_in = x.size(1);
    int H = x.size(2); int W = x.size(3);
    int C_out = weight.size(0); int K = weight.size(2);
    int OH = (H + 2 * padding - K) / stride + 1;
    int OW = (W + 2 * padding - K) / stride + 1;

    dim3 threads(16, 16);
    dim3 blocks((OW + 15) / 16, (OH + 15) / 16, N);
    
    fused_conv_min_tanh_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C_in, C_out, H, W, K, stride, padding, OH, OW);
}
"""

cpp_source = r"""
void fused_conv_min_tanh(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
                         torch::Tensor output, int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_tanh", &fused_conv_min_tanh, "Fused conv-min-tanh kernel");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    # conv_dilation and conv_groups are assumed to be default standard (1, 1) per prompt constraints
    N, C_in, H, W = x.shape
    C_out, _, K, _ = conv_weight.shape
    OH = (H + 2 * conv_padding - K) // conv_stride + 1
    OW = (W + 2 * conv_padding - K) // conv_stride + 1
    
    output = torch.empty((N, 1, OH, OW), device=x.device)
    fused_ext.fused_conv_min_tanh(x, conv_weight, conv_bias, output, conv_stride, conv_padding)
    return output
