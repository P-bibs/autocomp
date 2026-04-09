# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_100757/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a 3D convolution, applies minimum operation along a specific dimension, 
    and then applies softmax.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

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
    # State for conv (nn.Conv3d)
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
    if 'dim' in flat_state:
        state_kwargs['dim'] = flat_state['dim']
    else:
        state_kwargs['dim'] = getattr(model, 'dim')
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

# Define the CUDA kernel that fuses conv3d + min + softmax
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAMathCompat.h>

#define THREADS_PER_BLOCK 256

__global__ void fused_conv_min_softmax_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int D_in,
    const int H_in,
    const int W_in,
    const int D_out,
    const int H_out,
    const int W_out,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * H_out * W_out;
    
    if (idx >= total_elements) return;
    
    // Decompose index into components
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int out_ch = (idx / (W_out * H_out)) % out_channels;
    int batch = idx / (W_out * H_out * out_channels);
    
    // Calculate input positions
    int d_start = -padding;
    int d_end = D_in + padding - (kernel_size - 1) * dilation;
    
    float min_val = INFINITY;
    
    // Iterate through output depth dimension to find minimum
    for (int d_out = 0; d_out < D_out; ++d_out) {
        float accumulator = 0.0f;
        
        // Perform convolution at this position
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int d_in = d_out * stride + kd * dilation + d_start;
                    int h_in = h_out * stride + kh * dilation - padding;
                    int w_in = w_out * stride + kw * dilation - padding;
                    
                    // Check bounds
                    if (d_in >= 0 && d_in < D_in && 
                        h_in >= 0 && h_in < H_in && 
                        w_in >= 0 && w_in < W_in) {
                        
                        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
                            int input_idx = batch * (in_channels * D_in * H_in * W_in) +
                                          in_ch * (D_in * H_in * W_in) +
                                          d_in * (H_in * W_in) +
                                          h_in * W_in +
                                          w_in;
                                          
                            int weight_idx = out_ch * (in_channels * kernel_size * kernel_size * kernel_size) +
                                           in_ch * (kernel_size * kernel_size * kernel_size) +
                                           kd * (kernel_size * kernel_size) +
                                           kh * kernel_size +
                                           kw;
                                           
                            accumulator += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Add bias
        accumulator += bias[out_ch];
        
        // Update minimum
        if (accumulator < min_val) {
            min_val = accumulator;
        }
    }
    
    // Store the minimum value for softmax computation
    extern __shared__ float shared_mem[];
    float* shared_data = shared_mem;
    
    // Store min values in shared memory for softmax
    shared_data[threadIdx.x] = min_val;
    __syncthreads();
    
    // Compute softmax in block
    // First pass: find max
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < out_channels; i += blockDim.x) {
        if (shared_data[i] > max_val) max_val = shared_data[i];
    }
    
    // Reduction to find block max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && (threadIdx.x + stride) < out_channels) {
            float other = shared_data[threadIdx.x + stride];
            if (other > max_val) max_val = other;
        }
        __syncthreads();
    }
    
    // Second pass: compute sum of exp(x - max)
    float sum_exp = 0.0f;
    if (threadIdx.x < out_channels) {
        shared_data[threadIdx.x] = __expf(shared_data[threadIdx.x] - max_val);
        sum_exp = shared_data[threadIdx.x];
    }
    __syncthreads();
    
    // Reduction to find sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && (threadIdx.x + stride) < out_channels) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    sum_exp = shared_data[0];
    
    // Final pass: compute softmax
    if (threadIdx.x < out_channels) {
        float softmax_val = shared_data[threadIdx.x] / sum_exp;
        int out_idx = batch * (out_channels * H_out * W_out) +
                     out_ch * (H_out * W_out) +
                     h_out * W_out +
                     w_out;
        output[out_idx] = softmax_val;
    }
}

void fused_conv_min_softmax_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int stride,
    const int padding,
    const int dilation
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);
    
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2); // assuming cubic kernel
    
    const int D_out = (D_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int W_out = (W_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    const int total_elements = batch_size * out_channels * H_out * W_out;
    const int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Shared memory size for softmax computation
    const int shared_mem_size = out_channels * sizeof(float);
    
    fused_conv_min_softmax_kernel<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        kernel_size,
        stride,
        padding,
        dilation
    );
}
"""

# C++ source for PyBind11 binding
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_min_softmax_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int stride,
    const int padding,
    const int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_softmax_forward", &fused_conv_min_softmax_forward, "Fused Conv3D + Min + Softmax forward pass");
}
"""

# Compile the CUDA extension
fused_ext = load_inline(
    name='fused_conv_min_softmax',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Global variables for model parameters
batch_size = 128
in_channels = 3
out_channels = 24  # Increased output channels
D, H, W = 24, 32, 32  # Increased depth
kernel_size = 3
dim = 2  # Dimension along which to apply minimum operation (e.g., depth)
conv_stride = 1
conv_padding = 1
conv_dilation = 1
conv_groups = 1

# Store weights and biases as global tensors
conv_weight = None
conv_bias = None

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, W, H)]

def functional_model(
    x,
    *,
    conv_weight_param,
    conv_bias_param,
    conv_stride_param,
    conv_padding_param,
    conv_dilation_param,
    conv_groups_param,
    dim_param,
):
    global conv_weight, conv_bias
    
    # Store parameters if not already stored
    if conv_weight is None:
        conv_weight = conv_weight_param
        conv_bias = conv_bias_param
    
    # Calculate output dimensions
    D_out = (D + 2 * conv_padding_param - conv_dilation_param * (kernel_size - 1) - 1) // conv_stride_param + 1
    H_out = (H + 2 * conv_padding_param - conv_dilation_param * (kernel_size - 1) - 1) // conv_stride_param + 1
    W_out = (W + 2 * conv_padding_param - conv_dilation_param * (kernel_size - 1) - 1) // conv_stride_param + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_conv_min_softmax_forward(
        x, conv_weight, conv_bias, output,
        conv_stride_param, conv_padding_param, conv_dilation_param
    )
    
    return output
