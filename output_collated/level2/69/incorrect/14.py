# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_051208/code_2.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused convolution + hardswish + relu
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float hardswish_impl(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

__device__ float relu_impl(float x) {
    return fmaxf(x, 0.0f);
}

// Assumptions for this implementation:
// - Input tensor shape: [batch_size, in_channels, height, width]
// - Weight tensor shape: [out_channels, in_channels, kernel_h, kernel_w]
// - Bias tensor shape: [out_channels]
// - Convolution with stride=1, padding=1, dilation=1, groups=1 for simplicity
// This is a simplified version that matches the specific parameters in the original code.

__global__ void fused_conv_hardswish_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w
) {
    // Calculate output indices
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z;
    
    if (out_x >= width || out_y >= height || out_c >= out_channels) return;
    
    // Shared memory for weights (assuming kernel size is small)
    extern __shared__ float shared_weights[];
    
    float sum = 0.0f;
    
    // Load bias
    if (out_c < out_channels) {
        sum = bias[out_c];
    }
    
    // Perform convolution
    for (int in_c = 0; in_c < in_channels; ++in_c) {
        for (int ky = 0; ky < kernel_h; ++ky) {
            for (int kx = 0; kx < kernel_w; ++kx) {
                int in_x = out_x * stride_w - pad_w + kx * dilation_w;
                int in_y = out_y * stride_h - pad_h + ky * dilation_h;
                
                float val = 0.0f;
                if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                    val = input[((/*batch*/ 0 * in_channels + in_c) * height + in_y) * width + in_x];
                }
                
                float w_val = weight[((out_c * in_channels) + in_c) * kernel_h * kernel_w + ky * kernel_w + kx];
                sum += val * w_val;
            }
        }
    }
    
    // Apply activations: hardswish then relu
    float activated = relu_impl(hardswish_impl(sum));
    
    // Write output
    output[((/*batch*/ 0 * out_channels + out_c) * height + out_y) * width + out_x] = activated;
}

void launch_fused_conv_hardswish_relu(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    
    dim3 block(16, 16, 1);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y,
        out_channels
    );
    
    size_t shared_mem_size = kernel_h * kernel_w * in_channels * sizeof(float);
    
    fused_conv_hardswish_relu_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_h,
        kernel_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void launch_fused_conv_hardswish_relu(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_hardswish_relu", &launch_fused_conv_hardswish_relu, "Fused Convolution + HardSwish + ReLU");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_hardswish_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
):
    # Extract convolution parameters
    stride_h, stride_w = conv_stride if isinstance(conv_stride, (tuple, list)) else (conv_stride, conv_stride)
    pad_h, pad_w = conv_padding if isinstance(conv_padding, (tuple, list)) else (conv_padding, conv_padding)
    dilation_h, dilation_w = conv_dilation if isinstance(conv_dilation, (tuple, list)) else (conv_dilation, conv_dilation)
    
    # Create output tensor with correct shape
    out_channels = conv_weight.size(0)
    in_channels = conv_weight.size(1)
    kernel_h = conv_weight.size(2)
    kernel_w = conv_weight.size(3)
    
    # For simplicity, we assume batch_size=128 and fixed dimensions as in the original code
    batch_size, in_ch, height, width = x.shape
    out_height = (height + 2*pad_h - dilation_h*(kernel_h-1) - 1) // stride_h + 1
    out_width = (width + 2*pad_w - dilation_w*(kernel_w-1) - 1) // stride_w + 1
    
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Process each batch item individually since our kernel is simplified for batch_size=1
    for i in range(batch_size):
        x_single = x[i:i+1]  # Keep batch dimension
        output_single = output[i:i+1]
        
        fused_ext.fused_conv_hardswish_relu(
            x_single,
            conv_weight,
            conv_bias,
            output_single,
            kernel_h, kernel_w,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w
        )
    
    return output

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
