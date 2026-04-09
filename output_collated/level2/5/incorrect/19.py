# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_114641/code_2.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_kernel(float* __restrict__ data, const float* __restrict__ bias, int N, int C, int HW) {
    // Optimization 2: Use shared memory to cache bias
    extern __shared__ float shared_bias[];
    
    // Each thread in the first C threads loads one bias value
    if (threadIdx.x < C) {
        shared_bias[threadIdx.x] = bias[threadIdx.x];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C * HW) {
        int c = (idx / HW) % C;
        float val = data[idx];
        // Perform fused operation using cached bias
        data[idx] = tanhf(val - shared_bias[c]);
    }
}

void fused_op(torch::Tensor& x, const torch::Tensor& bias) {
    int N = x.size(0);
    int C = x.size(1);
    int HW = x.size(2) * x.size(3);
    int total_elements = N * C * HW;
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    // Shared memory size is C * sizeof(float)
    size_t shared_mem_size = C * sizeof(float);
    
    fused_op_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        N, C, HW
    );
}

__global__ void conv_transpose2d_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int input_height, int input_width, int kernel_size, int stride, 
    int padding, int output_padding, int dilation) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int output_elements_per_batch = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int total_output_elements = batch_size * out_channels * output_elements_per_batch * output_elements_per_batch;

    if (tid < total_output_elements) {
        int batch_idx = tid / (out_channels * output_elements_per_batch * output_elements_per_batch);
        int remaining = tid % (out_channels * output_elements_per_batch * output_elements_per_batch);
        int channel_out = remaining / (output_elements_per_batch * output_elements_per_batch);
        remaining = remaining % (output_elements_per_batch * output_elements_per_batch);
        int out_y = remaining / output_elements_per_batch;
        int out_x = remaining % output_elements_per_batch;

        float sum = 0.0f;
        for (int in_c = 0; in_c < in_channels; ++in_c) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int in_y = out_y + padding - ky * dilation;
                    int in_x = out_x + padding - kx * dilation;
                    
                    if (in_y % stride == 0 && in_x % stride == 0) {
                        in_y /= stride;
                        in_x /= stride;
                        
                        if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                            int input_idx = batch_idx * (in_channels * input_height * input_width) +
                                            in_c * (input_height * input_width) + 
                                            in_y * input_width + in_x;
                            int weight_idx = channel_out * (in_channels * kernel_size * kernel_size) +
                                             in_c * (kernel_size * kernel_size) + 
                                             ky * kernel_size + kx;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        output[tid] = sum + bias[channel_out];
    }
}

void launch_conv_transpose2d(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
    torch::Tensor& output, int kernel_size, int stride, int padding, int output_padding, int dilation) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);
    
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    int total_elements = batch_size * out_channels * output_height * output_width;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width, kernel_size, stride,
        padding, output_padding, dilation
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op(torch::Tensor& x, const torch::Tensor& bias);
void launch_conv_transpose2d(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
                             torch::Tensor& output, int kernel_size, int stride, int padding, int output_padding, int dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused Bias Subtraction and Tanh");
    m.def("conv_transpose2d", &launch_conv_transpose2d, "Custom Conv Transpose 2D");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
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
    # Only support groups=1 as we're implementing a simplified kernel
    if conv_transpose_groups != 1:
        raise ValueError("Custom conv transpose only supports groups=1")
        
    # Determine output dimensions
    batch_size, in_channels, input_height, input_width = x.shape
    out_channels, _, kernel_size, _ = conv_transpose_weight.shape
    
    # Compute output dimensions
    output_height = (input_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    output_width = (input_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, output_height, output_width), device=x.device, dtype=x.dtype)
    
    # Run custom convolution transpose
    fused_ext.conv_transpose2d(
        x, conv_transpose_weight, conv_transpose_bias, output,
        kernel_size, conv_transpose_stride, conv_transpose_padding, 
        conv_transpose_output_padding, conv_transpose_dilation[0]
    )
    
    # Flatten bias for kernel usage: bias_shape is (out_channels, 1, 1)
    bias_flat = bias.view(-1).contiguous()
    
    # Run custom fused kernel with shared memory optimization
    fused_ext.fused_op(output, bias_flat)
    
    return output

# Setup for testing
batch_size = 32
in_channels = 64
out_channels = 64
height = width = 32  # Reduced for demonstration, can be increased
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]
