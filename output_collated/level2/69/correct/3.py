# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_052229/code_1.py
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

__global__ void fused_conv2d_act_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W, int OC, int K,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w) {

    int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_oc = blockIdx.z;
    
    if (out_h >= (H + 2 * padding_h - dilation_h * (K - 1) - 1) / stride_h + 1 ||
        out_w >= (W + 2 * padding_w - dilation_w * (K - 1) - 1) / stride_w + 1)
        return;

    int batch_idx = batch_oc / OC;
    int oc_idx = batch_oc % OC;

    float acc = bias[oc_idx];
    
    for (int ic = 0; ic < C; ++ic) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int ih = out_h * stride_h - padding_h + kh * dilation_h;
                int iw = out_w * stride_w - padding_w + kw * dilation_w;
                
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float val = input[((batch_idx * C + ic) * H + ih) * W + iw];
                    float w = weight[((oc_idx * C + ic) * K + kh) * K + kw];
                    acc += val * w;
                }
            }
        }
    }
    
    // Hardswish(x) = x * relu6(x + 3) / 6
    float hswish = acc * fminf(fmaxf(acc + 3.0f, 0.0f), 6.0f) / 6.0f;
    // ReLU
    output[((batch_idx * OC + oc_idx) * (((H + 2 * padding_h - dilation_h * (K - 1) - 1) / stride_h + 1)) + out_h) * 
           (((W + 2 * padding_w - dilation_w * (K - 1) - 1) / stride_w + 1)) + out_w] = fmaxf(hswish, 0.0f);
}

void fused_conv2d_act(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                      int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w) {
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int OC = weight.size(0);
    int K = weight.size(2);
    
    int out_height = (H + 2 * padding_h - dilation_h * (K - 1) - 1) / stride_h + 1;
    int out_width = (W + 2 * padding_w - dilation_w * (K - 1) - 1) / stride_w + 1;
    
    dim3 block_size(16, 16);
    dim3 grid_size(
        (out_width + block_size.x - 1) / block_size.x,
        (out_height + block_size.y - 1) / block_size.y,
        N * OC
    );
    
    fused_conv2d_act_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C, H, W, OC, K,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv2d_act(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                      int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv2d_act", &fused_conv2d_act, "Fused Conv2d + Hardswish + ReLU");
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
    # Support for groups other than 1 is not implemented in this simplified version
    # This implementation assumes conv_groups=1
    
    # Handle different types of stride, padding, dilation
    if isinstance(conv_stride, int):
        stride_h, stride_w = conv_stride, conv_stride
    else:
        stride_h, stride_w = conv_stride[0], conv_stride[1]
        
    if isinstance(conv_padding, int):
        padding_h, padding_w = conv_padding, conv_padding
    else:
        padding_h, padding_w = conv_padding[0], conv_padding[1]
        
    if isinstance(conv_dilation, int):
        dilation_h, dilation_w = conv_dilation, conv_dilation
    else:
        dilation_h, dilation_w = conv_dilation[0], conv_dilation[1]
    
    out_height = (x.shape[2] + 2 * padding_h - dilation_h * (conv_weight.shape[2] - 1) - 1) // stride_h + 1
    out_width = (x.shape[3] + 2 * padding_w - dilation_w * (conv_weight.shape[3] - 1) - 1) // stride_w + 1
    out = torch.zeros((x.shape[0], conv_weight.shape[0], out_height, out_width), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_conv2d_act(x, conv_weight, conv_bias, out, 
                               stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w)
    return out
