# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_070444/code_4.py
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

# CUDA Kernel: Fused Batch/Channel Convolution
# Optimized to handle large feature maps by mapping 2D output dimensions to the grid.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int padding_h, int padding_w,
    int out_h, int out_w
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z;

    if (out_x >= out_w || out_y >= out_h || out_c >= out_channels) return;

    for (int b = 0; b < batch_size; ++b) {
        float sum = 0.0f;
        for (int in_c = 0; in_c < in_channels; ++in_c) {
            for (int ky = 0; ky < kernel_h; ++ky) {
                int in_y = out_y * stride_h - padding_h + ky;
                if (in_y < 0 || in_y >= height) continue;
                for (int kx = 0; kx < kernel_w; ++kx) {
                    int in_x = out_x * stride_w - padding_w + kx;
                    if (in_x >= 0 && in_x < width) {
                        sum += input[((b * in_channels + in_c) * height + in_y) * width + in_x] *
                               weight[((out_c * in_channels + in_c) * kernel_h + ky) * kernel_w + kx];
                    }
                }
            }
        }
        if (bias != nullptr) sum += bias[out_c];
        output[((b * out_channels + out_c) * out_h + out_y) * out_w + out_x] = sum;
    }
}

void conv2d_cuda(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, 
                 torch::Tensor& output, int sh, int sw, int ph, int pw) {
    int B = input.size(0), IC = input.size(1), H = input.size(2), W = input.size(3);
    int OC = weight.size(0), KH = weight.size(2), KW = weight.size(3);
    int OH = (H + 2 * ph - KH) / sh + 1;
    int OW = (W + 2 * pw - KW) / sw + 1;
    
    dim3 block(16, 16);
    dim3 grid((OW + 15) / 16, (OH + 15) / 16, OC);
    
    conv2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr, output.data_ptr<float>(),
        B, IC, OC, H, W, KH, KW, sh, sw, ph, pw, OH, OW
    );
}
"""

cpp_source = r"""
void conv2d_cuda(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& output, int sh, int sw, int ph, int pw);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("conv2d", &conv2d_cuda); }
"""

conv_ext = load_inline(name='conv_ext', cpp_sources=cpp_source, cuda_sources=cuda_kernel, with_cuda=True)

def functional_model(x, *, conv1d_weight, conv1d_bias, conv1d_stride, conv1d_padding, conv1d_dilation, conv1d_groups):
    # Functional forward pass using custom CUDA kernel
    sh = sw = conv1d_stride
    ph = pw = conv1d_padding
    
    out_h = (x.size(2) + 2 * ph - conv1d_weight.size(2)) // sh + 1
    out_w = (x.size(3) + 2 * pw - conv1d_weight.size(3)) // sw + 1
    output = torch.empty((x.size(0), conv1d_weight.size(0), out_h, out_w), device='cuda')
    
    conv_ext.conv2d(x, conv1d_weight, conv1d_bias, output, sh, sw, ph, pw)
    return output
