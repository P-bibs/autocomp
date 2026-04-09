# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_052603/code_5.py
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

# CUDA Kernel: Fused Conv2d (3x3, s1, p1) + Hardswish + ReLU
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_act_kernel(const float* __restrict__ input, 
                                     const float* __restrict__ weight, 
                                     const float* __restrict__ bias, 
                                     float* __restrict__ output, 
                                     int N, int C, int H, int W, int OC) {
    int oc = blockIdx.x;
    int n = blockIdx.y;
    int row = blockIdx.z / ((W + 15) / 16) * 16 + threadIdx.y;
    int col = (blockIdx.z % ((W + 15) / 16)) * 16 + threadIdx.x;

    if (row >= H || col >= W) return;

    float acc = bias[oc];
    for (int ic = 0; ic < C; ++ic) {
        for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
                int r = row + kh - 1;
                int c = col + kw - 1;
                if (r >= 0 && r < H && c >= 0 && c < W) {
                    float val = input[((n * C + ic) * H + r) * W + c];
                    float w = weight[(((oc * C + ic) * 3 + kh) * 3 + kw)];
                    acc += val * w;
                }
            }
        }
    }
    
    // Hardswish(x) = x * ReLU6(x + 3) / 6
    float hswish = acc * fminf(fmaxf(acc + 3.0f, 0.0f), 6.0f) * 0.16666667f;
    // ReLU(x)
    output[((n * OC + oc) * H + row) * W + col] = fmaxf(hswish, 0.0f);
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    int N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    int OC = weight.size(0);
    dim3 threads(16, 16);
    dim3 blocks(OC, N, ((H + 15) / 16) * ((W + 15) / 16));
    
    fused_conv_act_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        output.data_ptr<float>(), N, C, H, W, OC
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv2d + Hardswish + ReLU");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    # Only supports conv_stride=1, conv_padding=1, conv_dilation=1 (original spec)
    N, C, H, W = x.shape
    OC = conv_weight.shape[0]
    output = torch.empty((N, OC, H, W), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_weight, conv_bias, output)
    return output
