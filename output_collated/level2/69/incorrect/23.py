# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_051539/code_6.py
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

# CUDA kernel for fused conv2d + hardswish + relu using implicit GEMM logic
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fused_conv_hardswish_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int IC, int OC, int IH, int IW, 
    int KH, int KW, int SH, int SW, 
    int PH, int PW, int DH, int DW, 
    int G, int OH, int OW
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * OC * OH * OW;
    if (tid >= total_elements) return;

    // Index mapping
    int tmp = tid;
    int ow = tmp % OW; tmp /= OW;
    int oh = tmp % OH; tmp /= OH;
    int oc = tmp % OC; tmp /= OC;
    int b = tmp;

    int g = oc / (OC / G);
    int ic_per_g = IC / G;
    int oc_per_g = OC / G;

    float sum = 0.0f;
    for (int kh = 0; kh < KH; ++kh) {
        for (int kw = 0; kw < KW; ++kw) {
            int ih = oh * SH + kh * DH - PH;
            int iw = ow * SW + kw * DW - PW;
            if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                for (int ic = 0; ic < ic_per_g; ++ic) {
                    int input_idx = ((b * IC + g * ic_per_g + ic) * IH + ih) * IW + iw;
                    int weight_idx = (oc * ic_per_g + ic) * (KH * KW) + (kh * KW + kw);
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    if (bias != nullptr) sum += bias[oc];

    // Fused Hardswish (x * min(max(x + 3, 0), 6) / 6) and ReLU (max(x, 0))
    // Note: Applying ReLU after Hardswish is redundant if Hardswish output is >= 0,
    // but we strictly follow the logic requested.
    float hs = sum + 3.0f;
    if (hs < 0.0f) hs = 0.0f;
    else if (hs > 6.0f) hs = 6.0f;
    float res = sum * hs / 6.0f;
    
    output[tid] = (res > 0.0f) ? res : 0.0f;
}

void fused_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride, int padding, int dilation, int groups
) {
    int B = input.size(0);
    int IC = input.size(1);
    int IH = input.size(2);
    int IW = input.size(3);
    int OC = weight.size(0);
    int KH = weight.size(2);
    int KW = weight.size(3);
    int OH = output.size(2);
    int OW = output.size(3);

    int total_threads = B * OC * OH * OW;
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    fused_conv_hardswish_relu_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.size(0) > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B, IC, OC, IH, IW, KH, KW, stride, stride, padding, padding, dilation, dilation, groups, OH, OW
    );
}
"""

cpp_source = r"""
void fused_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward", &fused_forward, "Fused conv+hs+relu");
}
"""

fused_ext = load_inline(
    name='fused_conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    B = x.shape[0]
    OC = conv_weight.shape[0]
    H_out = (x.shape[2] + 2 * conv_padding - conv_dilation * (conv_weight.shape[2] - 1) - 1) // conv_stride + 1
    W_out = (x.shape[3] + 2 * conv_padding - conv_dilation * (conv_weight.shape[3] - 1) - 1) // conv_stride + 1
    
    output = torch.empty((B, OC, H_out, W_out), device=x.device, dtype=x.dtype)
    fused_ext.fused_forward(
        x.contiguous(), conv_weight.contiguous(), 
        conv_bias if conv_bias is not None else torch.tensor([], device=x.device),
        output, conv_stride, conv_padding, conv_dilation, conv_groups
    )
    return output

batch_size, in_channels, out_channels, height, width, kernel_size = 128, 8, 64, 128, 128, 3
def get_init_inputs(): return [in_channels, out_channels, kernel_size]
def get_inputs(): return [torch.rand(batch_size, in_channels, height, width)]
