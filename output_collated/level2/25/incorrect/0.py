# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_081239/code_2.py
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

# Define the CUDA kernel for fused operation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAMathCompat.h>

__global__ void fused_op_forward_kernel(
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
    int dilation,
    int groups
) {
    // Calculate output dimensions
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Get thread indices
    int batch_idx = blockIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (batch_idx >= batch_size || out_y >= out_height || out_x >= out_width) return;
    
    // Shared memory for weights (assuming small kernel size)
    extern __shared__ float shared_weight[];
    
    // Load weights into shared memory
    int weights_per_thread = (out_channels * in_channels * kernel_size * kernel_size + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < weights_per_thread; i++) {
        int idx = threadIdx.x + i * blockDim.x;
        if (idx < out_channels * in_channels * kernel_size * kernel_size) {
            shared_weight[idx] = weight[idx];
        }
    }
    __syncthreads();
    
    // Perform convolution with min reduction across channels
    float min_val = INFINITY;
    
    // For each output channel
    for (int out_c = 0; out_c < out_channels; out_c++) {
        float sum = 0.0f;
        
        // For each group
        int group_idx = out_c / (out_channels / groups);
        int in_channels_per_group = in_channels / groups;
        int start_in_channel = group_idx * in_channels_per_group;
        int end_in_channel = start_in_channel + in_channels_per_group;
        
        // Convolution operation
        for (int in_c = start_in_channel; in_c < end_in_channel; in_c++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    // Calculate input position
                    int in_y = out_y * stride - padding + ky * dilation;
                    int in_x = out_x * stride - padding + kx * dilation;
                    
                    // Check bounds
                    if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                        int input_idx = batch_idx * (in_channels * height * width) + 
                                       in_c * (height * width) + 
                                       in_y * width + in_x;
                        int weight_idx = out_c * (in_channels * kernel_size * kernel_size) + 
                                        in_c * (kernel_size * kernel_size) + 
                                        ky * kernel_size + kx;
                        
                        sum += input[input_idx] * shared_weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias
        sum += bias[out_c];
        
        // Update minimum
        if (sum < min_val) {
            min_val = sum;
        }
    }
    
    // Apply tanh twice
    min_val = tanhf(min_val);
    min_val = tanhf(min_val);
    
    // Write output
    int output_idx = batch_idx * (1 * out_height * out_width) + 
                    0 * (out_height * out_width) + 
                    out_y * out_width + out_x;
    output[output_idx] = min_val;
}

void fused_op_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    // Get tensor dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    // Calculate output dimensions
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Define block and grid dimensions
    dim3 block_size(32, 4, 4); // 32 threads for weights, 4x4 for spatial
    dim3 grid_size(batch_size, (out_height + block_size.y - 1) / block_size.y, 
                   (out_width + block_size.z - 1) / block_size.z);
    
    // Shared memory size for weights
    int shared_mem_size = out_channels * in_channels * kernel_size * kernel_size * sizeof(float);
    
    // Launch kernel
    fused_op_forward_kernel<<<grid_size, block_size, shared_mem_size>>>(
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
        groups
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}
"""

# Define C++ interface
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv2D + Min + Tanh operation");
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
    # Create output tensor with correct shape
    # Output has 1 channel after min reduction
    batch_size = x.size(0)
    in_channels = x.size(1)
    height = x.size(2)
    width = x.size(3)
    kernel_size = conv_weight.size(2)
    
    out_height = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    output = torch.empty(batch_size, 1, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call the fused operation
    fused_ext.fused_op(x, conv_weight, conv_bias, output, conv_stride, conv_padding, conv_dilation, conv_groups)
    
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
