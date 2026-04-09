# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_085904/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
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

# CUDA kernel that fuses conv2d + min + tanh + tanh operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAMathCompat.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// Better approach: process spatial locations and compute min across channels
__global__ void fused_conv_min_tanh_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {
    
    // Calculate output dimensions
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Grid-stride loop over spatial locations
    int spatial_size = batch_size * out_height * out_width;
    CUDA_1D_KERNEL_LOOP(spatial_idx, spatial_size) {
        // Calculate spatial indices
        int tmp = spatial_idx;
        int out_w = tmp % out_width; tmp /= out_width;
        int out_h = tmp % out_height; tmp /= out_height;
        int batch = tmp;
        
        // Calculate input starting position
        int in_h_start = out_h * stride - padding;
        int in_w_start = out_w * stride - padding;
        
        // Compute convolutions for all output channels at this spatial location
        float min_val = 1e30f; // Large initial value
        
        for (int out_c = 0; out_c < out_channels; ++out_c) {
            float conv_result = 0.0f;
            
            // Convolution loop
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int in_h = in_h_start + kh * dilation;
                    int in_w = in_w_start + kw * dilation;
                    
                    // Check bounds
                    if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                        for (int ic = 0; ic < in_channels; ++ic) {
                            // Calculate weight index
                            int weight_idx = out_c * (in_channels * kernel_size * kernel_size) + 
                                           ic * (kernel_size * kernel_size) + 
                                           kh * kernel_size + kw;
                            
                            // Calculate input index
                            int input_idx = batch * (in_channels * height * width) + 
                                          ic * (height * width) + 
                                          in_h * width + in_w;
                            
                            conv_result += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            
            // Add bias
            conv_result += bias[out_c];
            
            // Track minimum
            if (conv_result < min_val) {
                min_val = conv_result;
            }
        }
        
        // Apply tanh twice to the minimum value
        float result = tanhf(min_val);
        result = tanhf(result);
        
        // Store result (keepdim=True means we keep the channel dimension as 1)
        int output_idx = batch * out_height * out_width + out_h * out_width + out_w;
        output[output_idx] = result;
    }
}

void fused_conv_min_tanh_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int dilation) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Calculate spatial size for grid-stride loop
    int spatial_size = batch_size * out_height * out_width;
    
    // Launch configuration
    const int threads_per_block = 256;
    const int blocks = (spatial_size + threads_per_block - 1) / threads_per_block;
    
    fused_conv_min_tanh_kernel<<<blocks, threads_per_block>>>(
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
        dilation
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ Logic (Interface/Bindings)
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_min_tanh_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_tanh", &fused_conv_min_tanh_forward, "Fused Conv2D + Min + Tanh + Tanh operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
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
    # Calculate output dimensions for the final result
    batch_size = x.shape[0]
    out_height = (x.shape[2] + 2 * conv_padding - conv_dilation * (conv_weight.shape[2] - 1) - 1) // conv_stride + 1
    out_width = (x.shape[3] + 2 * conv_padding - conv_dilation * (conv_weight.shape[3] - 1) - 1) // conv_stride + 1
    
    # Create output tensor with correct shape (keepdim=True for min operation)
    output = torch.empty(batch_size, 1, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call the fused CUDA kernel
    fused_ext.fused_conv_min_tanh(
        x, conv_weight, conv_bias, output,
        conv_stride, conv_padding, conv_dilation
    )
    
    return output

batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
