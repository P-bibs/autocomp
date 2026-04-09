# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_051208/code_4.py
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

# CUDA kernel that fuses convolution, hardswish, and relu
# Uses 2D grid/blocks to optimize memory coalescing and occupancy
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_act_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int out_h,
    int out_w
) {
    int out_n = blockIdx.z; // Batch index
    int out_c = blockIdx.y; // Channel index
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x; // Spatial index (h * out_w + w)

    if (out_n >= batch_size || out_c >= out_channels || out_idx >= out_h * out_w) return;

    int out_y = out_idx / out_w;
    int out_x = out_idx % out_w;

    float sum = 0.0f;
    int weight_base = out_c * (in_channels * kernel_size * kernel_size);
    int input_base = out_n * (in_channels * height * width);

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int iy = out_y * stride - padding + ky * dilation;
                int ix = out_x * stride - padding + kx * dilation;

                if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                    float val = input[input_base + ic * (height * width) + iy * width + ix];
                    float w = weight[weight_base + ic * (kernel_size * kernel_size) + ky * kernel_size + kx];
                    sum += val * w;
                }
            }
        }
    }

    sum += bias[out_c];

    // Hardswish(x) = x * ReLU6(x + 3) / 6
    float hs = sum * fminf(fmaxf(sum + 3.0f, 0.0f), 6.0f) / 6.0f;
    // ReLU(x)
    float res = fmaxf(hs, 0.0f);

    output[out_n * (out_channels * out_h * out_w) + out_c * (out_h * out_w) + out_idx] = res;
}

void fused_conv_act_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int dilation
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    dim3 block(256);
    dim3 grid((out_h * out_w + 255) / 256, out_channels, batch_size);
    
    fused_conv_act_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        dilation,
        out_h,
        out_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_conv_act_forward(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& output, int stride, int padding, int dilation);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_act", &fused_conv_act_forward, "Fused convolution and activations");
}
"""

fused_ext = load_inline(
    name='fused_conv_act',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    # This implementation assumes groups=1 as per typical optimized kernels for this task
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_weight.shape[0]
    kernel_size = conv_weight.shape[2]
    
    out_height = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_conv_act(x, conv_weight, conv_bias, output, conv_stride, conv_padding, conv_dilation)
    return output
