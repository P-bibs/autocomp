# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_072125/code_6.py
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

# Tiled CUDA kernel for 2D Convolution to replace F.conv2d
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int ic, int ih, int iw,
    int oc, int kh, int kw,
    int stride, int padding
) {
    int out_h = (ih + 2 * padding - kh) / stride + 1;
    int out_w = (iw + 2 * padding - kw) / stride + 1;
    
    int oc_idx = blockIdx.x; 
    int batch_idx = blockIdx.y;
    int out_hw = threadIdx.x + blockIdx.z * blockDim.x;
    
    if (oc_idx >= oc || batch_idx >= batch || out_hw >= out_h * out_w) return;
    
    int oh = out_hw / out_w;
    int ow = out_hw % out_w;
    
    float val = (bias != nullptr) ? bias[oc_idx] : 0.0f;
    
    for (int c = 0; c < ic; ++c) {
        for (int i = 0; i < kh; ++i) {
            for (int j = 0; j < kw; ++j) {
                int h_in = oh * stride + i - padding;
                int w_in = ow * stride + j - padding;
                
                if (h_in >= 0 && h_in < ih && w_in >= 0 && w_in < iw) {
                    float in_val = input[((batch_idx * ic + c) * ih + h_in) * iw + w_in];
                    float w_val = weight[((oc_idx * ic + c) * kh + i) * kw + j];
                    val += in_val * w_val;
                }
            }
        }
    }
    output[((batch_idx * oc + oc_idx) * out_h + oh) * out_w + ow] = val;
}

void launch_conv2d(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride, int padding
) {
    int batch = input.size(0);
    int ic = input.size(1);
    int ih = input.size(2);
    int iw = input.size(3);
    int oc = weight.size(0);
    int kh = weight.size(2);
    int kw = weight.size(3);
    int oh = output.size(2);
    int ow = output.size(3);

    dim3 threads(256);
    dim3 blocks(oc, batch, (oh * ow + 255) / 256);
    
    conv2d_tiled_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.size(0) > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch, ic, ih, iw, oc, kh, kw, stride, padding
    );
}
"""

cpp_source = r"""
void launch_conv2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("conv2d", &launch_conv2d, "Custom Conv2D"); }
"""

conv_ext = load_inline(
    name='custom_conv', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'], with_cuda=True
)

def functional_model(x, *, conv1d_weight, conv1d_bias, conv1d_stride, conv1d_padding, conv1d_dilation, conv1d_groups):
    # dilation is assumed 1 for this kernel implementation
    batch, ic, ih, iw = x.shape
    oc, _, kh, kw = conv1d_weight.shape
    oh = (ih + 2 * conv1d_padding - kh) // conv1d_stride + 1
    ow = (iw + 2 * conv1d_padding - kw) // conv1d_stride + 1
    
    out = torch.empty((batch, oc, oh, ow), device=x.device, dtype=x.dtype)
    conv_ext.conv2d(x, conv1d_weight, conv1d_bias, out, conv1d_stride, conv1d_padding)
    return out
