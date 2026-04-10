# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_065607/code_4.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv1d_weight', 'conv1d_bias', 'conv1d_stride', 'conv1d_padding', 'conv1d_dilation', 'conv1d_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv1d_weight', 'conv1d_bias']


class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool=False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

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
    # State for conv1d (nn.Conv2d)
    if 'conv1d_weight' in flat_state:
        state_kwargs['conv1d_weight'] = flat_state['conv1d_weight']
    else:
        state_kwargs['conv1d_weight'] = getattr(model.conv1d, 'weight', None)
    if 'conv1d_bias' in flat_state:
        state_kwargs['conv1d_bias'] = flat_state['conv1d_bias']
    else:
        state_kwargs['conv1d_bias'] = getattr(model.conv1d, 'bias', None)
    state_kwargs['conv1d_stride'] = model.conv1d.stride
    state_kwargs['conv1d_padding'] = model.conv1d.padding
    state_kwargs['conv1d_dilation'] = model.conv1d.dilation
    state_kwargs['conv1d_groups'] = model.conv1d.groups
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

# CUDA kernel: Tiled 2D convolution
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C, int H, int W,
    int OC, int KH, int KW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int OH, int OW
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_oc = blockIdx.z;
    int b = batch_oc / OC;
    int oc = batch_oc % OC;

    if (ow < OW && oh < OH) {
        float val = bias[oc];
        for (int ic = 0; ic < C; ++ic) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int ih = oh * stride_h - pad_h + kh;
                    int iw = ow * stride_w - pad_w + kw;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        val += input[((b * C + ic) * H + ih) * W + iw] * 
                               weight[((oc * C + ic) * KH + kh) * KW + kw];
                    }
                }
            }
        }
        output[((b * OC + oc) * OH + oh) * OW + ow] = val;
    }
}

void cuda_conv2d(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride_h, int stride_w, int pad_h, int pad_w
) {
    int B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    int OC = weight.size(0), KH = weight.size(2), KW = weight.size(3);
    int OH = output.size(2), OW = output.size(3);

    dim3 threads(16, 16);
    dim3 blocks((OW + 15) / 16, (OH + 15) / 16, B * OC);

    conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        B, C, H, W, OC, KH, KW, stride_h, stride_w, pad_h, pad_w, OH, OW
    );
}
"""

cpp_source = r"""
void cuda_conv2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int s_h, int s_w, int p_h, int p_w);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d", &cuda_conv2d, "Custom CUDA Conv2D");
}
"""

conv_ext = load_inline(
    name='conv_ext', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv1d_weight, conv1d_bias, conv1d_stride, conv1d_padding, conv1d_dilation, conv1d_groups):
    # Ensure inputs are on GPU
    device = torch.device('cuda')
    x = x.to(device)
    w = conv1d_weight.to(device)
    b = conv1d_bias.to(device)
    
    # Standardize parameters (assuming 2D emulation of convolution)
    s = conv1d_stride if isinstance(conv1d_stride, tuple) else (conv1d_stride, conv1d_stride)
    p = conv1d_padding if isinstance(conv1d_padding, tuple) else (conv1d_padding, conv1d_padding)
    
    B, C, H, W = x.shape
    OC, _, KH, KW = w.shape
    OH = (H + 2 * p[0] - KH) // s[0] + 1
    OW = (W + 2 * p[1] - KW) // s[1] + 1
    
    out = torch.empty((B, OC, OH, OW), device=device, dtype=x.dtype)
    conv_ext.conv2d(x, w, b, out, s[0], s[1], p[0], p[1])
    return out
