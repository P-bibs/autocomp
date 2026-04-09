# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_051902/code_5.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused kernel: Conv2d(3x3) + Hardswish + ReLU
// Optimized for throughput using vector-like accumulation in registers
__global__ void fused_conv_hardswish_relu_kernel(
    const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
    float* __restrict__ out, int N, int C, int H, int W, int OC) {
    
    int n = blockIdx.z;
    int oc = blockIdx.x;
    int h = blockIdx.y / W;
    int w_idx = blockIdx.y % W;

    // Accumulate sum in a register
    float acc = b[oc];
    
    // Perform 3x3 Conv accumulation
    // Padding=1, Stride=1 implied by logic
    for (int ic = 0; ic < C; ++ic) {
        for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
                int ih = h + kh - 1;
                int iw = w_idx + kw - 1;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float val = x[((n * C + ic) * H + ih) * W + iw];
                    float weight = w[((oc * C + ic) * 3 + kh) * 3 + kw];
                    acc += val * weight;
                }
            }
        }
    }

    // Hardswish: x * relu6(x + 3) / 6
    float hs = acc * fminf(fmaxf(acc + 3.0f, 0.0f), 6.0f) / 6.0f;
    // ReLU
    out[((n * OC + oc) * H + h) * W + w_idx] = fmaxf(hs, 0.0f);
}

void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out) {
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int OC = weight.size(0);
    
    // Grid: (OutChannels, Height * Width, BatchSize)
    dim3 blocks(OC, H * W, N);
    fused_conv_hardswish_relu_kernel<<<blocks, 1>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), N, C, H, W, OC);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv/Hardswish/ReLU kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    # Functional forward pass using fused CUDA kernel
    # Assumes stride=1, padding=1, dilation=1 for 3x3 conv
    out = torch.empty((x.shape[0], conv_weight.shape[0], x.shape[2], x.shape[3]), device=x.device)
    fused_ext.fused_op(x.contiguous(), conv_weight.contiguous(), conv_bias.contiguous(), out)
    return out
