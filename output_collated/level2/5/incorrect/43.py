# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_121229/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'bias_shape', 'stride', 'padding', 'output_padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

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
    # State for conv_transpose (nn.ConvTranspose2d)
    if 'conv_transpose_weight' in flat_state:
        state_kwargs['conv_transpose_weight'] = flat_state['conv_transpose_weight']
    else:
        state_kwargs['conv_transpose_weight'] = getattr(model.conv_transpose, 'weight', None)
    if 'conv_transpose_bias' in flat_state:
        state_kwargs['conv_transpose_bias'] = flat_state['conv_transpose_bias']
    else:
        state_kwargs['conv_transpose_bias'] = getattr(model.conv_transpose, 'bias', None)
    state_kwargs['conv_transpose_stride'] = model.conv_transpose.stride
    state_kwargs['conv_transpose_padding'] = model.conv_transpose.padding
    state_kwargs['conv_transpose_output_padding'] = model.conv_transpose.output_padding
    state_kwargs['conv_transpose_groups'] = model.conv_transpose.groups
    state_kwargs['conv_transpose_dilation'] = model.conv_transpose.dilation
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
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

# CUDA Kernel: Custom conv transpose + fused bias subtraction and Tanh
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_THREAD 4

// Vectorized fused operation kernel
__global__ void fused_op_kernel_vectorized(float* __restrict__ data, 
                                          const float* __restrict__ bias, 
                                          int N, int C, int HW) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;
    
    if (idx + ELEMENTS_PER_THREAD <= N * C * HW) {
        // Vectorized load using float4
        float4* data4 = (float4*)data;
        float4 vals = data4[idx / 4];
        
        // Process 4 elements at once
        int base_idx = idx;
        int c0 = (base_idx / HW) % C;
        vals.x = tanhf(vals.x - bias[c0]);
        
        int c1 = ((base_idx + 1) / HW) % C;
        vals.y = tanhf(vals.y - bias[c1]);
        
        int c2 = ((base_idx + 2) / HW) % C;
        vals.z = tanhf(vals.z - bias[c2]);
        
        int c3 = ((base_idx + 3) / HW) % C;
        vals.w = tanhf(vals.w - bias[c3]);
        
        data4[idx / 4] = vals;
    } else {
        // Handle remaining elements
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            int actual_idx = idx + i;
            if (actual_idx < N * C * HW) {
                int c = (actual_idx / HW) % C;
                float val = data[actual_idx];
                data[actual_idx] = tanhf(val - bias[c]);
            }
        }
    }
}

// Optimized convolution transpose 2D kernel
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
) {
    // Calculate output dimensions
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * out_height * out_width;
    
    if (tid >= total_output_elements) return;
    
    // Calculate indices
    int batch_idx = tid / (out_channels * out_height * out_width);
    int channel_idx = (tid / (out_height * out_width)) % out_channels;
    int out_y = (tid / out_width) % out_height;
    int out_x = tid % out_width;
    
    float sum = 0.0f;
    
    // Perform convolution transpose operation
    int group = channel_idx / (out_channels / groups);
    int weight_base_c = channel_idx * (in_channels / groups) * kernel_size * kernel_size;
    int input_base_n = batch_idx * in_channels * in_height * in_width;
    
    for (int c = 0; c < in_channels / groups; c++) {
        int input_c = c + group * (in_channels / groups);
        if (input_c >= in_channels) continue;
        
        int input_base_c = input_base_n + input_c * in_height * in_width;
        int weight_base = weight_base_c + c * kernel_size * kernel_size;
        
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                // Calculate corresponding input position
                int in_y = (out_y + padding - ky * dilation) / stride;
                int in_x = (out_x + padding - kx * dilation) / stride;
                
                // Check bounds and stride alignment
                if (in_y >= 0 && in_y < in_height && 
                    in_x >= 0 && in_x < in_width &&
                    (out_y + padding - ky * dilation) % stride == 0 &&
                    (out_x + padding - kx * dilation) % stride == 0) {
                    
                    int input_idx = input_base_c + in_y * in_width + in_x;
                    int weight_idx = weight_base + ky * kernel_size + kx;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    output[tid] = sum + bias[channel_idx];
}

void fused_op_forward(torch::Tensor& x, const torch::Tensor& bias) {
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int HW = H * W;
    int total_elements = N * C * HW;
    
    int threads = THREADS_PER_BLOCK;
    int blocks = (total_elements + threads * ELEMENTS_PER_THREAD - 1) / (threads * ELEMENTS_PER_THREAD);
    
    fused_op_kernel_vectorized<<<blocks, threads>>>(
        x.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        N, C, HW
    );
}

void conv_transpose2d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    int total_elements = batch_size * out_channels * out_height * out_width;
    int threads = THREADS_PER_BLOCK;
    int blocks = (total_elements + threads - 1) / threads;
    
    conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        dilation
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor& x, const torch::Tensor& bias);
void conv_transpose2d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused operation kernel");
    m.def("conv_transpose2d", &conv_transpose2d_forward, "Conv transpose 2D kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    bias,
):
    # Create output tensor with correct shape
    batch_size = x.size(0)
    in_channels = x.size(1)
    in_height = x.size(2)
    in_width = x.size(3)
    out_channels = conv_transpose_weight.size(0)
    kernel_size = conv_transpose_weight.size(2)
    
    # Calculate output dimensions for conv transpose
    out_height = (in_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Perform custom convolution transpose
    fused_ext.conv_transpose2d(
        x, conv_transpose_weight, conv_transpose_bias, output,
        conv_transpose_stride, conv_transpose_padding, 
        conv_transpose_output_padding, conv_transpose_groups, 
        conv_transpose_dilation
    )
    
    # Flatten bias for kernel usage: bias_shape is (out_channels, 1, 1)
    bias_flat = bias.view(-1)
    
    # Run custom fused kernel to perform (x - bias) and tanh in one pass
    fused_ext.fused_op(output, bias_flat)
    
    return output

# Setup for testing
batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]
