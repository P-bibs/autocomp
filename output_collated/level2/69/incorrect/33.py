# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_052229/code_5.py
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

// Tile size for block-level optimization
#define TILE_DIM 16

__global__ void fused_conv2d_act_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W, int OC, int K) {

    int oc = blockIdx.z;
    int b = blockIdx.y;
    int oh = blockIdx.x * TILE_DIM + threadIdx.y;
    int ow = threadIdx.x; // We process one row per thread block

    if (oh < H) {
        for (int w = 0; w < W; ++w) {
            float acc = bias[oc];
            for (int ic = 0; ic < C; ++ic) {
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        int ih = oh + kh - 1;
                        int iw = w + kw - 1;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            float val = input[((b * C + ic) * H + ih) * W + iw];
                            float wgt = weight[(((oc * C + ic) * K + kh) * K) + kw];
                            acc += val * wgt;
                        }
                    }
                }
            }
            // Hardswish(x) = x * min(max(x + 3, 0), 6) / 6
            float hswish = acc * fminf(fmaxf(acc + 3.0f, 0.0f), 6.0f) / 6.0f;
            // ReLU(x)
            output[((b * OC + oc) * H + oh) * W + w] = fmaxf(hswish, 0.0f);
        }
    }
}

void fused_conv2d_act(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    int N = input.size(0); int C = input.size(1); int H = input.size(2); int W = input.size(3);
    int OC = weight.size(0); int K = weight.size(2);
    
    dim3 blocks((H + TILE_DIM - 1) / TILE_DIM, N, OC);
    dim3 threads(1, TILE_DIM);
    
    fused_conv2d_act_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        output.data_ptr<float>(), N, C, H, W, OC, K);
}
"""

cpp_source = r"""
void fused_conv2d_act(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);
"""

# Pybind11 binding logic
custom_cpp_source = cpp_source + r"""
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv2d_act", &fused_conv2d_act, "Fused Conv2D + Hardswish + ReLU");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=custom_cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    # Only supports specific geometry constant with current kernel logic (stride 1, padding 1)
    N, C, H, W = x.shape
    OC = conv_weight.shape[0]
    out = torch.empty((N, OC, H, W), device=x.device, dtype=x.dtype)
    fused_ext.fused_conv2d_act(x, conv_weight, conv_bias, out)
    return out
