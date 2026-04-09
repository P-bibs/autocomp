# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_115905/code_6.py
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

# CUDA kernel for fused bias subtraction and tanh activation with vectorization
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Vectorized kernel with shared memory optimization
__global__ void fused_bias_tanh_kernel_vectorized(float* __restrict__ data, 
                                                  const float* __restrict__ bias, 
                                                  int N, int C, int H, int W) {
    int total_elements = N * C * H * W;
    int hw = H * W;
    
    // Shared memory for bias values
    extern __shared__ float shared_bias[];
    
    // Collaboratively load bias into shared memory
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        shared_bias[i] = bias[i];
    }
    __syncthreads();
    
    // Vectorized processing: each thread handles 4 elements
    int elements_per_thread = 4;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_work_units = (total_elements + elements_per_thread - 1) / elements_per_thread;
    
    for (int work_id = thread_idx; work_id < total_work_units; work_id += gridDim.x * blockDim.x) {
        int start_idx = work_id * elements_per_thread;
        
        #pragma unroll
        for (int offset = 0; offset < elements_per_thread; offset++) {
            int i = start_idx + offset;
            if (i < total_elements) {
                // Compute channel index for bias lookup
                int c = (i / hw) % C;
                
                // Load, compute, and store back
                float val = data[i];
                val = val - shared_bias[c];
                data[i] = tanhf(val);
            }
        }
    }
}

// Kernel for custom convolution transpose (simplified version for performance)
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding, int output_padding
) {
    // Compute output dimensions
    int kernel_radius = kernel_size / 2;
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    int total_output_elements = batch_size * out_channels * out_height * out_width;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (thread_idx >= total_output_elements) return;
    
    // Decompose linear index into 4D indices
    int temp = thread_idx;
    int w_out = temp % out_width; temp /= out_width;
    int h_out = temp % out_height; temp /= out_height;
    int c_out = temp % out_channels; temp /= out_channels;
    int n = temp;
    
    float sum = 0.0f;
    
    // Determine input region that contributes to this output element
    // Based on transpose convolution formula
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Map output coordinate back to input space
                int h_in = (h_out + padding - kh);
                int w_in = (w_out + padding - kw);
                
                // Check if it's within valid input bounds and stride condition
                if (h_in >= 0 && h_in < in_height * stride && h_in % stride == 0 &&
                    w_in >= 0 && w_in < in_width * stride && w_in % stride == 0) {
                    
                    h_in /= stride;
                    w_in /= stride;
                    
                    if (h_in < in_height && w_in < in_width) {
                        int input_idx = n * (in_channels * in_height * in_width) +
                                        c_in * (in_height * in_width) +
                                        h_in * in_width + w_in;
                                        
                        int weight_idx = c_out * (in_channels * kernel_size * kernel_size) +
                                         c_in * (kernel_size * kernel_size) +
                                         kh * kernel_size + kw;
                                         
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add bias and write to output
    output[thread_idx] = sum + bias[c_out];
}

void fused_op_forward(torch::Tensor x, torch::Tensor bias, int optimal_threads) {
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    
    int total_elements = N * C * H * W;
    int elements_per_thread = 4;
    int total_work_units = (total_elements + elements_per_thread - 1) / elements_per_thread;
    
    // Use provided optimal thread count
    int threads = optimal_threads;
    int blocks = (total_work_units + threads - 1) / threads;
    blocks = min(blocks, 65535);
    
    // Shared memory size for bias
    size_t shared_mem_size = C * sizeof(float);
    
    fused_bias_tanh_kernel_vectorized<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        N, C, H, W
    );
    
    cudaDeviceSynchronize();
}

void conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding,
    int optimal_threads
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2); // Assuming square kernel
    
    int out_height = output.size(2);
    int out_width = output.size(3);
    
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    // Launch configuration
    int threads = optimal_threads;
    int blocks = (total_elements + threads - 1) / threads;
    blocks = min(blocks, 65535);
    
    conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_size, stride, padding, output_padding
    );
    
    cudaDeviceSynchronize();
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor x, torch::Tensor bias, int optimal_threads);
void conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_tensor,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding,
    int optimal_threads
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Bias Subtraction and Tanh with Vectorization");
    m.def("conv_transpose2d_op", &conv_transpose2d_forward, "Custom Conv Transpose 2D");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def get_optimal_thread_count(device_id=0):
    """
    Determine optimal thread count for RTX 2080Ti.
    """
    return 512

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
    """
    Optimized functional model without using built-in PyTorch conv/matmul functions.
    """
    # Assumptions for simplification:
    # - conv_transpose_groups = 1 (no group conv)
    # - conv_transpose_dilation = 1 (no dilation)
    # - kernel is square
    
    # 1. Custom Convolution Transpose using our CUDA kernel
    batch_size = x.size(0)
    in_channels = x.size(1)
    in_height = x.size(2)
    in_width = x.size(3)
    
    out_channels = conv_transpose_weight.size(0)
    kernel_size = conv_transpose_weight.size(2)
    
    # Compute output dimensions
    out_height = (in_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Get optimal thread count
    optimal_threads = get_optimal_thread_count()
    
    # Launch custom convolution
    fused_ext.conv_transpose2d_op(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        output,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        optimal_threads
    )
    
    # 2. Fused operation using optimized vectorized CUDA kernel
    bias_flat = bias.view(-1).contiguous()
    output = output.contiguous()
    
    # Call the optimized kernel
    fused_ext.fused_op(output, bias_flat, optimal_threads)
    
    return output

# Placeholder parameters
batch_size = 32
in_channels = 64  
out_channels = 64  
height = width = 256 
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
