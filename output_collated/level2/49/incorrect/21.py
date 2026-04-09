# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_094451/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'softmax_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

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
    # State for conv_transpose (nn.ConvTranspose3d)
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
    # State for softmax (nn.Softmax)
    state_kwargs['softmax_dim'] = model.softmax.dim
    # State for sigmoid (nn.Sigmoid)
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

# Custom CUDA kernel for fused conv_transpose3d + softmax + sigmoid
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cmath>

#define MAX_THREADS_PER_BLOCK 1024

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    if (idx >= total_elements) return;
    
    // Decompose linear index to 5D coordinates
    int temp = idx;
    int w_out = temp % output_w;
    temp /= output_w;
    int h_out = temp % output_h;
    temp /= output_h;
    int d_out = temp % output_d;
    temp /= output_d;
    int c_out = temp % out_channels;
    int n = temp / out_channels;
    
    // Calculate input position
    float sum = (bias) ? bias[c_out] : 0.0f;
    
    // Loop through kernel
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Calculate corresponding input position
                int d_in = d_out + padding - kd;
                int h_in = h_out + padding - kh;
                int w_in = w_out + padding - kw;
                
                // Check if within input bounds after accounting for stride
                if (d_in % stride == 0 && h_in % stride == 0 && w_in % stride == 0) {
                    d_in /= stride;
                    h_in /= stride;
                    w_in /= stride;
                    
                    if (d_in >= 0 && d_in < input_d &&
                        h_in >= 0 && h_in < input_h &&
                        w_in >= 0 && w_in < input_w) {
                        
                        // Calculate indices
                        int input_idx = n * (in_channels * input_d * input_h * input_w) +
                                       0 * (input_d * input_h * input_w) +  // channel index will be looped
                                       d_in * (input_h * input_w) +
                                       h_in * input_w +
                                       w_in;
                                       
                        int weight_idx = c_out * (in_channels * kernel_size * kernel_size * kernel_size) +
                                        0 * (kernel_size * kernel_size * kernel_size) + // channel index will be looped
                                        kd * (kernel_size * kernel_size) +
                                        kh * kernel_size +
                                        kw;
                                        
                        // Loop through input channels
                        for (int c_in = 0; c_in < in_channels; c_in++) {
                            int final_input_idx = input_idx + c_in * (input_d * input_h * input_w);
                            int final_weight_idx = weight_idx + c_in * (kernel_size * kernel_size * kernel_size);
                            sum += input[final_input_idx] * weight[final_weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    output[idx] = sum;
}

__global__ void fused_softmax_sigmoid_kernel(
    const float* input,
    float* output,
    int total_elements,
    int softmax_dim_size,
    int inner_dim_size
) {
    // Handle softmax and sigmoid in one kernel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Calculate which softmax group this element belongs to
    int softmax_group = idx / inner_dim_size;
    int pos_in_group = idx % inner_dim_size;
    int group_start = softmax_group * softmax_dim_size;
    
    // Shared memory for softmax computation
    extern __shared__ float shared_data[];
    float* softmax_data = shared_data;
    
    // Load data into shared memory
    if (threadIdx.x < softmax_dim_size) {
        softmax_data[threadIdx.x] = input[group_start + threadIdx.x * inner_dim_size + pos_in_group];
    }
    __syncthreads();
    
    // Find max for numerical stability
    float max_val = -INFINITY;
    if (threadIdx.x < softmax_dim_size) {
        max_val = softmax_data[threadIdx.x];
    }
    
    for (int stride = softmax_dim_size / 2; stride > 0; stride >>= 1) {
        float temp = -INFINITY;
        if (threadIdx.x < stride && threadIdx.x + stride < softmax_dim_size) {
            temp = softmax_data[threadIdx.x + stride];
        }
        max_val = fmaxf(max_val, temp);
        __syncthreads();
    }
    
    // Compute exponentials and sum
    float exp_val = 0.0f;
    if (threadIdx.x < softmax_dim_size) {
        exp_val = expf(softmax_data[threadIdx.x] - max_val);
        softmax_data[threadIdx.x] = exp_val;
    }
    __syncthreads();
    
    float sum_exp = 0.0f;
    if (threadIdx.x < softmax_dim_size) {
        sum_exp = softmax_data[threadIdx.x];
    }
    
    for (int stride = softmax_dim_size / 2; stride > 0; stride >>= 1) {
        float temp = 0.0f;
        if (threadIdx.x < stride && threadIdx.x + stride < softmax_dim_size) {
            temp = softmax_data[threadIdx.x + stride];
        }
        sum_exp += temp;
        __syncthreads();
    }
    
    // Apply softmax and sigmoid
    if (threadIdx.x < softmax_dim_size) {
        float softmax_result = softmax_data[threadIdx.x] / sum_exp;
        float sigmoid_result = 1.0f / (1.0f + expf(-softmax_result));
        output[group_start + threadIdx.x * inner_dim_size + pos_in_group] = sigmoid_result;
    }
}

void fused_model_forward(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor output,
    int softmax_dim,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    // Launch conv_transpose3d kernel
    int total_output_elements = batch_size * out_channels * output_d * output_h * output_w;
    int threads_per_block = min(MAX_THREADS_PER_BLOCK, 1024);
    int blocks_for_conv = (total_output_elements + threads_per_block - 1) / threads_per_block;
    
    conv_transpose3d_kernel<<<blocks_for_conv, threads_per_block>>>(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_size,
        stride,
        padding,
        output_padding
    );
    
    cudaDeviceSynchronize();
    
    // Launch fused softmax + sigmoid kernel
    int inner_dim_size, softmax_dim_size, reduce_dim_size;
    
    if (softmax_dim == 1) { // Channel dimension
        softmax_dim_size = out_channels;
        inner_dim_size = output_d * output_h * output_w;
        reduce_dim_size = batch_size;
    } else if (softmax_dim == 2) { // D dimension
        softmax_dim_size = output_d;
        inner_dim_size = output_h * output_w;
        reduce_dim_size = batch_size * out_channels;
    } else if (softmax_dim == 3) { // H dimension
        softmax_dim_size = output_h;
        inner_dim_size = output_w;
        reduce_dim_size = batch_size * out_channels * output_d;
    } else if (softmax_dim == 4) { // W dimension
        softmax_dim_size = output_w;
        inner_dim_size = 1;
        reduce_dim_size = batch_size * out_channels * output_d * output_h;
    } else {
        return; // Invalid dimension
    }
    
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    int blocks_for_fuse = (total_elements + threads_per_block - 1) / threads_per_block;
    size_t shared_mem_size = softmax_dim_size * sizeof(float);
    
    fused_softmax_sigmoid_kernel<<<blocks_for_fuse, threads_per_block, shared_mem_size>>>(
        output.data_ptr<float>(),
        output.data_ptr<float>(), // In-place operation
        total_elements,
        softmax_dim_size,
        inner_dim_size
    );
}
"""

# C++ source for bindings
cpp_source = r"""
#include <torch/extension.h>

void fused_model_forward(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor output,
    int softmax_dim,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_model_forward", &fused_model_forward, "Fused ConvTranspose3d + Softmax + Sigmoid forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_model_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
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
    softmax_dim,
):
    # Extract dimensions
    batch_size, in_channels, D, H, W = x.shape
    out_channels = conv_transpose_weight.shape[1]  # Note: for conv_transpose, weight is (in_channels, out_channels/groups, ...)
    
    # Calculate output dimensions for conv_transpose3d
    kernel_size = conv_transpose_weight.shape[2]  # Assuming cubic kernel
    output_d = (D - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + kernel_size + conv_transpose_output_padding[0]
    output_h = (H - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + kernel_size + conv_transpose_output_padding[1]
    output_w = (W - 1) * conv_transpose_stride[2] - 2 * conv_transpose_padding[2] + kernel_size + conv_transpose_output_padding[2]
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, output_d, output_h, output_w), device=x.device, dtype=x.dtype)
    
    # Call fused CUDA implementation
    fused_ext.fused_model_forward(
        x.contiguous(),
        conv_transpose_weight.contiguous(),
        conv_transpose_bias.contiguous(),
        output,
        softmax_dim,
        batch_size,
        in_channels,
        out_channels,
        D, H, W,
        output_d, output_h, output_w,
        kernel_size,
        conv_transpose_stride[0],
        conv_transpose_padding[0],
        conv_transpose_output_padding[0]
    )
    
    return output

# Constants
batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]
