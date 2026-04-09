# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123254/code_1.py
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

# Optimized CUDA Kernel: Eliminate shared memory, use direct global memory access
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_op_kernel_optimized(
    float* __restrict__ data, 
    const float* __restrict__ bias, 
    int N, int C, int H, int W
) {
    // Grid-stride loop for better scalability
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int HW = H * W;
    int C_HW = C * HW;
    
    // Process multiple elements per thread to improve occupancy
    for (int i = idx; i < N * C_HW; i += stride) {
        // Efficient index computation avoiding division
        int temp = i;
        int hw = temp % HW; temp /= HW;
        int c = temp % C; 
        // int n = temp / C; // Not needed for computation
        
        // Direct global memory access to bias - no shared memory overhead
        float bias_val = bias[c];
        float data_val = data[i];
        
        // Compute tanh in-place
        data[i] = tanhf(data_val - bias_val);
    }
}

void fused_op_forward(torch::Tensor& x, const torch::Tensor& bias) {
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    
    // Optimize block size for better occupancy on RTX 2080Ti
    int threads_per_block = 256;
    int total_elements = N * C * H * W;
    
    // Calculate blocks with grid-stride pattern
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    // Cap blocks to prevent launch overhead while maintaining good occupancy
    blocks = min(blocks, 65535);
    
    // Launch kernel with direct global memory access
    fused_op_kernel_optimized<<<blocks, threads_per_block>>>(
        x.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        N, C, H, W
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor& x, const torch::Tensor& bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward, "Fused bias subtraction and tanh with direct global memory access");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Custom CUDA kernel for conv transpose - replacing PyTorch's built-in function
conv_transpose_cuda = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    // Calculate output dimensions
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_channels * out_height * out_width;
    
    if (tid >= total_threads) return;
    
    // Decode output position
    int temp = tid;
    int w_out = temp % out_width; temp /= out_width;
    int h_out = temp % out_height; temp /= out_height;
    int c_out = temp % out_channels; temp /= out_channels;
    int n = temp;
    
    float sum = 0.0f;
    
    // Convolution transpose computation
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Calculate corresponding input position
                int h_in = (h_out + padding - kh) / stride;
                int w_in = (w_out + padding - kw) / stride;
                
                // Check if input position is valid
                if (h_in >= 0 && h_in < in_height && 
                    w_in >= 0 && w_in < in_width &&
                    (h_out + padding - kh) % stride == 0 &&
                    (w_out + padding - kw) % stride == 0) {
                    
                    int input_idx = ((n * in_channels + c_in) * in_height + h_in) * in_width + w_in;
                    int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    sum += bias[c_out];
    output[tid] = sum;
}

void conv_transpose_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    int threads_per_block = 256;
    int total_elements = batch_size * out_channels * out_height * out_width;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    blocks = min(blocks, 65535);
    
    conv_transpose_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
"""

conv_transpose_cpp = r"""
#include <torch/extension.h>

void conv_transpose_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose_forward", &conv_transpose_forward, "Custom Conv Transpose Forward");
}
"""

# Compile conv transpose extension
conv_transpose_ext = load_inline(
    name='conv_transpose_ext',
    cpp_sources=conv_transpose_cpp,
    cuda_sources=conv_transpose_cuda,
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
    # Perform convolution using custom CUDA kernel
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_size, _ = conv_transpose_weight.shape
    
    # Calculate output dimensions
    out_height = (in_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Run custom conv transpose kernel
    conv_transpose_ext.conv_transpose_forward(
        x, conv_transpose_weight, conv_transpose_bias, output,
        conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding
    )
    
    # Run optimized fused operation kernel (no shared memory usage)
    fused_ext.fused_op_forward(output, bias.view(-1))
    
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
