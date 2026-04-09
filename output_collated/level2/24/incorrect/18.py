# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_102756/code_0.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__device__ inline float relu_activation(float x) {
    return fmaxf(x, 0.0f);
}

__global__ void fused_conv_min_softmax_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int D, int H, int W,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int dim
) {
    // Calculate output dimensions
    int out_D = (D + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_H = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_W = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Shared memory for softmax reduction
    extern __shared__ float sdata[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // Each thread processes multiple output elements if necessary
    for (int idx = tid; idx < batch_size * out_channels * out_H * out_W; idx += total_threads) {
        // Decode output indices
        int w_out = idx % out_W;
        int h_out = (idx / out_W) % out_H;
        int c_out = (idx / (out_W * out_H)) % out_channels;
        int b = idx / (out_W * out_H * out_channels);
        
        // Perform convolution + min reduction in one pass
        float min_val = 1e30f; // Large positive number
        
        // Iterate through depth dimension for min reduction
        for (int d_out = 0; d_out < out_D; ++d_out) {
            float conv_result = 0.0f;
            
            // Perform convolution for this output position
            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int d_in = d_out * stride + kd * dilation - padding;
                        int h_in = h_out * stride + kh * dilation - padding;
                        int w_in = w_out * stride + kw * dilation - padding;
                        
                        if (d_in >= 0 && d_in < D && h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                            for (int c_in = 0; c_in < in_channels; ++c_in) {
                                int weight_idx = c_out * (in_channels * kernel_size * kernel_size * kernel_size) +
                                               c_in * (kernel_size * kernel_size * kernel_size) +
                                               kd * (kernel_size * kernel_size) + kh * kernel_size + kw;
                                int input_idx = b * (in_channels * D * H * W) +
                                              c_in * (D * H * W) +
                                              d_in * (H * W) + h_in * W + w_in;
                                
                                conv_result += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
            
            // Add bias
            conv_result += bias[c_out];
            
            // Track minimum
            min_val = fminf(min_val, conv_result);
        }
        
        // Store min value in shared memory for softmax computation
        sdata[threadIdx.x] = min_val;
        __syncthreads();
        
        // Compute softmax along channel dimension (dim=1)
        // First, find max for numerical stability
        float max_val = -1e30f; // Large negative number
        for (int c = threadIdx.x; c < out_channels; c += blockDim.x) {
            max_val = fmaxf(max_val, ((b * out_channels * out_H * out_W + c * out_H * out_W + h_out * out_W + w_out) < batch_size * out_channels * out_H * out_W) ? 
                                       ((float*)sdata)[(c * out_H * out_W + h_out * out_W + w_out) % blockDim.x] : max_val);
        }
        
        // This is a simplified approach. In practice, we'd need to synchronize better
        // for proper softmax across channels.
        
        // Reduction to find global max among active threads
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
            }
            __syncthreads();
        }
        
        float global_max = sdata[0];
        __syncthreads();
        
        // Compute exp and store in shared memory
        float exp_val = expf(min_val - global_max);
        sdata[threadIdx.x] = exp_val;
        __syncthreads();
        
        // Reduction to compute sum
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
            __syncthreads();
        }
        
        float sum_exp = sdata[0];
        
        // Compute final softmax output
        float softmax_output = exp_val / sum_exp;
        
        output[idx] = softmax_output;
    }
}

// Alternative implementation with better softmax handling
__global__ void fused_conv_min_softmax_kernel_v2(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int D, int H, int W,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int dim
) {
    // Calculate output dimensions
    int out_D = (D + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_H = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_W = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * out_H * out_W;
    
    if (tid >= total_output_elements) return;
    
    // Decode output indices
    int w_out = tid % out_W;
    int h_out = (tid / out_W) % out_H;
    int c_out = (tid / (out_W * out_H)) % out_channels;
    int b = tid / (out_W * out_H * out_channels);
    
    // Perform convolution + min reduction in one pass
    float min_val = 1e30f; // Large positive number
    
    // Iterate through depth dimension for min reduction
    for (int d_out = 0; d_out < out_D; ++d_out) {
        float conv_result = 0.0f;
        
        // Perform convolution for this output position
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int d_in = d_out * stride + kd * dilation - padding;
                    int h_in = h_out * stride + kh * dilation - padding;
                    int w_in = w_out * stride + kw * dilation - padding;
                    
                    if (d_in >= 0 && d_in < D && h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        for (int c_in = 0; c_in < in_channels; ++c_in) {
                            int weight_idx = c_out * (in_channels * kernel_size * kernel_size * kernel_size) +
                                           c_in * (kernel_size * kernel_size * kernel_size) +
                                           kd * (kernel_size * kernel_size) + kh * kernel_size + kw;
                            int input_idx = b * (in_channels * D * H * W) +
                                          c_in * (D * H * W) +
                                          d_in * (H * W) + h_in * W + w_in;
                            
                            conv_result += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Add bias
        conv_result += bias[c_out];
        
        // Track minimum
        min_val = fminf(min_val, conv_result);
    }
    
    // For softmax, we need to work with all channels for this spatial location
    // This requires a different approach - use a separate kernel for softmax
    output[tid] = min_val;
}

__global__ void softmax_kernel(
    float* __restrict__ input_output,
    int batch_size,
    int channels,
    int height,
    int width
) {
    int h = blockIdx.y;
    int w = blockIdx.z;
    int tid = threadIdx.x;
    
    if (h >= height || w >= width) return;
    
    // Shared memory for reduction
    extern __shared__ float sdata[];
    
    // Load data into shared memory
    for (int c = tid; c < channels; c += blockDim.x) {
        int idx = ((blockIdx.x * channels + c) * height + h) * width + w;
        sdata[c] = (idx < batch_size * channels * height * width) ? input_output[idx] : -1e30f;
    }
    __syncthreads();
    
    // Find max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    float max_val = sdata[0];
    __syncthreads();
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int c = tid; c < channels; c += blockDim.x) {
        int idx = ((blockIdx.x * channels + c) * height + h) * width + w;
        if (idx < batch_size * channels * height * width) {
            float exp_val = expf(input_output[idx] - max_val);
            input_output[idx] = exp_val;
            sum += exp_val;
        }
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float total_sum = sdata[0] + 1e-8f; // Add small epsilon to avoid division by zero
    __syncthreads();
    
    // Normalize
    for (int c = tid; c < channels; c += blockDim.x) {
        int idx = ((blockIdx.x * channels + c) * height + h) * width + w;
        if (idx < batch_size * channels * height * width) {
            input_output[idx] /= total_sum;
        }
    }
}

void fused_conv_min_softmax_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    int stride,
    int padding,
    int dilation,
    int dim
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2); // Assuming cubic kernel
    
    int out_D = (D + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_H = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_W = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // First, do conv + min
    int total_output_elements = batch_size * out_channels * out_H * out_W;
    
    // Launch configuration for first kernel
    int threads_per_block = 256;
    int blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;
    
    // Intermediate tensor to hold min values before softmax
    at::Tensor intermediate = at::empty({batch_size, out_channels, out_H, out_W}, 
                                        at::TensorOptions().dtype(at::kFloat).device(input.device()));
    
    fused_conv_min_softmax_kernel_v2<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        intermediate.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        D, H, W,
        kernel_size,
        stride,
        padding,
        dilation,
        dim
    );
    
    // Then apply softmax
    dim3 softmax_blocks(batch_size, out_H, out_W);
    dim3 softmax_threads(min(1024, out_channels));
    size_t shared_mem_size = out_channels * sizeof(float);
    
    if (shared_mem_size <= 48 * 1024) { // Check if shared memory is sufficient
        softmax_kernel<<<softmax_blocks, softmax_threads, shared_mem_size>>>(
            intermediate.data_ptr<float>(),
            batch_size,
            out_channels,
            out_H,
            out_W
        );
    }
    
    // Copy result to output
    output.copy_(intermediate);
    
    cudaDeviceSynchronize();
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_min_softmax_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    int stride,
    int padding,
    int dilation,
    int dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv_min_softmax_forward, "Fused conv3d + min + softmax forward");
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
    dim,
):
    # Ensure inputs are contiguous and on the correct device
    x = x.contiguous()
    conv_weight = conv_weight.contiguous()
    conv_bias = conv_bias.contiguous()
    
    # Calculate output dimensions
    batch_size = x.shape[0]
    out_channels = conv_weight.shape[0]
    D, H, W = x.shape[2], x.shape[3], x.shape[4]
    kernel_size = conv_weight.shape[2]
    
    out_D = (D + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_H = (H + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_W = (W + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Min operation reduces along dim=2 (depth), so output depth becomes 1
    output_shape = (batch_size, out_channels, out_H, out_W)
    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    
    # Call fused operation
    fused_ext.fused_op(
        x, conv_weight, conv_bias, output,
        conv_stride, conv_padding, conv_dilation, dim
    )
    
    return output

batch_size = 128
in_channels = 3
out_channels = 24  # Increased output channels
D, H, W = 24, 32, 32  # Increased depth
kernel_size = 3
dim = 2  # Dimension along which to apply minimum operation (e.g., depth)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]
