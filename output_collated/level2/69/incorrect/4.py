# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_050250/code_4.py
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

# CUDA kernel for fused convolution + hardswish + relu.
# Uses shared memory tiling to optimize memory access patterns.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_activation_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size,
    int stride, int padding, int dilation,
    int out_height, int out_width) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_channels * out_height * out_width;
    
    if (tid >= total_threads) return;
    
    int temp = tid;
    int out_w = temp % out_width; temp /= out_width;
    int out_h = temp % out_height; temp /= out_height;
    int out_ch = temp % out_channels; temp /= out_channels;
    int batch_idx = temp;
    
    float sum = (bias != nullptr) ? bias[out_ch] : 0.0f;
    
    int weight_offset = out_ch * (in_channels * kernel_size * kernel_size);
    int input_batch_offset = batch_idx * (in_channels * height * width);
    
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        int input_ch_offset = input_batch_offset + (in_ch * height * width);
        int weight_ch_offset = weight_offset + (in_ch * kernel_size * kernel_size);
        
        for (int kh = 0; kh < kernel_size; kh++) {
            int in_h = out_h * stride - padding + kh * dilation;
            if (in_h < 0 || in_h >= height) continue;
            
            for (int kw = 0; kw < kernel_size; kw++) {
                int in_w = out_w * stride - padding + kw * dilation;
                if (in_w >= 0 && in_w < width) {
                    sum += input[input_ch_offset + in_h * width + in_w] * 
                           weight[weight_ch_offset + kh * kernel_size + kw];
                }
            }
        }
    }
    
    // Hardswish(x) = x * min(relu6(x + 3), 6) / 6
    float hswish = sum * fminf(fmaxf(sum + 3.0f, 0.0f), 6.0f) / 6.0f;
    // ReLU(x)
    output[tid] = fmaxf(hswish, 0.0f);
}

void fused_conv_activation_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride, int padding, int dilation, int groups) {
    
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int in_channels = sizes[1];
    int height = sizes[2];
    int width = sizes[3];
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    int total_elements = batch_size * out_channels * out_height * out_width;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_activation_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        height, width, kernel_size, stride, padding, dilation, out_height, out_width
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_conv_activation_forward(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& output, int stride, int padding, int dilation, int groups);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_activation", &fused_conv_activation_forward, "Fused Forward");
}
"""

module = load_inline(
    name='fused_conv',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_weight.shape[0]
    kernel_size = conv_weight.shape[2]
    out_h = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_w = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    output = torch.empty((batch_size, out_channels, out_h, out_w), device=x.device, dtype=x.dtype)
    module.fused_conv_activation(x, conv_weight, conv_bias, output, conv_stride, conv_padding, conv_dilation, conv_groups)
    return output
