# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_051539/code_7.py
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

# ----------------------------------------------------------------------
# CUDA source – fused conv + hardswish + relu kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_act_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N,
    const int batch,
    const int in_ch,
    const int out_ch,
    const int in_h,
    const int in_w,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int out_h,
    const int out_w)
{
    // Grid-stride loop
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
        int tmp = idx;
        int out_c = tmp % out_ch;
        tmp /= out_ch;
        int out_y = tmp % out_h;
        int out_x = tmp % out_w;
        int b = tmp / out_w;

        float sum = 0.0f;
        int in_ch_per_group = in_ch / groups;
        int group_id = out_c / (out_ch / groups);
        int in_ch_start = group_id * in_ch_per_group;

        for (int ic = 0; ic < in_ch_per_group; ++ic) {
            int ic_global = in_ch_start + ic;
            for (int ky = 0; ky < kernel_size; ++ky) {
                int iy = out_y * stride - padding + ky * dilation;
                if (iy < 0 || iy >= in_h) continue;
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int ix = out_x * stride - padding + kx * dilation;
                    if (ix < 0 || ix >= in_w) continue;

                    int weight_idx = ((out_c * in_ch_per_group) + ic) * (kernel_size * kernel_size)
                                     + ky * kernel_size + kx;
                    int input_idx = ((b * in_ch + ic_global) * in_h + iy) * in_w + ix;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }

        if (bias != nullptr) sum += bias[out_c];

        // Activation: Hardswish(x) = x * ReLU6(x+3) / 6, then ReLU
        float hs = sum * fminf(fmaxf(sum + 3.0f, 0.0f), 6.0f) / 6.0f;
        output[idx] = fmaxf(0.0f, hs);
    }
}

void fused_conv_act(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int batch, int in_ch, int out_ch, int in_h, int in_w,
    int kernel_size, int stride, int padding, int dilation, int groups,
    int out_h, int out_w, torch::Tensor output)
{
    int N = batch * out_ch * out_h * out_w;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 4096) blocks = 4096;

    fused_conv_act_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), N, batch, in_ch, out_ch, in_h, in_w,
        kernel_size, stride, padding, dilation, groups, out_h, out_w);
}
"""

cpp_source = r"""
void fused_conv_act(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                    int batch, int in_ch, int out_ch, int in_h, int in_w,
                    int kernel_size, int stride, int padding, int dilation, int groups,
                    int out_h, int out_w, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_act", &fused_conv_act, "Fused Conv/Hardswish/ReLU");
}
"""

fused_ext = load_inline(
    name='fused_conv_act', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    x, conv_weight = x.cuda(), conv_weight.cuda()
    conv_bias = conv_bias.cuda() if conv_bias is not None else torch.tensor([], device='cuda')
    batch, in_ch, in_h, in_w = x.shape
    out_ch, _, k_size, _ = conv_weight.shape
    out_h = (in_h + 2 * conv_padding - conv_dilation * (k_size - 1) - 1) // conv_stride + 1
    out_w = (in_w + 2 * conv_padding - conv_dilation * (k_size - 1) - 1) // conv_stride + 1
    output = torch.empty((batch, out_ch, out_h, out_w), dtype=x.dtype, device='cuda')
    fused_ext.fused_conv_act(x, conv_weight, conv_bias, batch, in_ch, out_ch, in_h, in_w,
                            k_size, conv_stride, conv_padding, conv_dilation, conv_groups,
                            out_h, out_w, output)
    return output
