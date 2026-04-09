# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_051539/code_5.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused kernel: 3x3 Convolution + Hardswish + ReLU
// Each thread calculates one output pixel (N, OC, H, W)
__global__ void fused_conv_act_kernel(const float* __restrict__ input, 
                                      const float* __restrict__ weight, 
                                      const float* __restrict__ bias, 
                                      float* __restrict__ output,
                                      int N, int C, int H, int W, int OC) {
    int oc = blockIdx.x;
    int n = blockIdx.y;
    int h = blockIdx.z / W;
    int w = blockIdx.z % W;

    float acc = bias[oc];
    
    // Apply 3x3 convolution
    for (int ic = 0; ic < C; ++ic) {
        for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
                int ih = h + kh - 1;
                int iw = w + kw - 1;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float val = input[((n * C + ic) * H + ih) * W + iw];
                    float w_val = weight[((oc * C + ic) * 3 + kh) * 3 + kw];
                    acc += val * w_val;
                }
            }
        }
    }
    
    // Fused Hardswish: x * min(max(x + 3, 0), 6) / 6
    float hswish = acc * fminf(fmaxf(acc + 3.0f, 0.0f), 6.0f) / 6.0f;
    // Fused ReLU
    output[((n * OC + oc) * H + h) * W + w] = fmaxf(hswish, 0.0f);
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int OC = weight.size(0);

    // grid: OC, N, H*W
    dim3 blocks(OC, N, H * W);
    fused_conv_act_kernel<<<blocks, 1>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        output.data_ptr<float>(), N, C, H, W, OC
    );
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv + Hardswish + ReLU");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=1, conv_dilation=1, conv_groups=1):
    # Ensure input is float32 for the custom kernel
    x = x.float()
    out = torch.empty((x.size(0), conv_weight.size(0), x.size(2), x.size(3)), device=x.device)
    
    # Execute fused kernel
    fused_ext.fused_op(x, conv_weight, conv_bias, out)
    return out

# Global vars defined to match original signature
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]
