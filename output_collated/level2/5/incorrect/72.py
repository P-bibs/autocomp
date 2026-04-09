# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123254/code_9.py
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

# CUDA Kernel: Winograd F(4x4, 3x3) optimized implementation with fused bias+tanh
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Winograd F(4x4, 3x3) implementation
// Input tile size: 6x6, Output tile size: 4x4
// Precomputed Winograd transformation matrices
__constant__ float BT[36] = {
    4.0f, 0.0f, -5.0f, 0.0f, 1.0f, 0.0f,
    0.0f, -4.0f, -4.0f, 1.0f, 1.0f, 0.0f,
    0.0f, 4.0f, -4.0f, -1.0f, 1.0f, 0.0f,
    0.0f, -2.0f, -1.0f, 2.0f, 1.0f, 0.0f,
    0.0f, 2.0f, -1.0f, -2.0f, 1.0f, 0.0f,
    0.0f, 4.0f, 0.0f, -5.0f, 0.0f, 1.0f
};

__constant__ float B[36] = {
    4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, -4.0f, 4.0f, -2.0f, 2.0f, 4.0f,
    -5.0f, -4.0f, -4.0f, -1.0f, -1.0f, 0.0f,
    0.0f, 1.0f, -1.0f, 2.0f, -2.0f, -5.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f
};

__constant__ float G[18] = {
    1.0f, 0.0f, 0.0f,
    0.5f, 0.5f, 0.5f,
    0.5f, -0.5f, 0.5f,
    0.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f
};

__constant__ float GT[18] = {
    1.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.5f, -0.5f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f
};

__device__ void winograd_transform_input(float* input_tile, float* transformed_tile) {
    float temp[36];
    
    // BT * d
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 6; k++) {
                sum += BT[i * 6 + k] * input_tile[k * 6 + j];
            }
            temp[i * 6 + j] = sum;
        }
    }
    
    // (BT * d) * B
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 6; k++) {
                sum += temp[i * 6 + k] * B[k * 6 + j];
            }
            transformed_tile[i * 6 + j] = sum;
        }
    }
}

__device__ void winograd_transform_kernel(float* kernel, float* transformed_kernel) {
    float temp[18];
    
    // G * g
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += G[i * 3 + k] * kernel[k * 3 + j];
            }
            temp[i * 3 + j] = sum;
        }
    }
    
    // (G * g) * GT
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += temp[i * 3 + k] * GT[k * 6 + j];
            }
            transformed_kernel[i * 6 + j] = sum;
        }
    }
}

__device__ void winograd_multiply(float* U, float* V, float* M) {
    for (int i = 0; i < 36; i++) {
        M[i] = U[i] * V[i];
    }
}

__device__ void winograd_transform_output(float* M, float* output_tile) {
    float temp[36];
    
    // AT * M
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 6; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 6; k++) {
                sum += BT[i * 6 + k] * M[k * 6 + j];  // BT is also AT for this specific matrix
            }
            temp[i * 6 + j] = sum;
        }
    }
    
    // (AT * M) * A
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 6; k++) {
                sum += temp[i * 6 + k] * B[k * 6 + j];  // B is also A for this specific matrix
            }
            output_tile[i * 4 + j] = sum;
        }
    }
}

__global__ void winograd_conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out, int H_in, int W_in, int H_out, int W_out,
    int kernel_size
) {
    // Calculate tile indices
    int batch_idx = blockIdx.x;
    int channel_out_idx = blockIdx.y;
    int tile_idx = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (batch_idx >= N || channel_out_idx >= C_out) return;
    
    // Precompute constants
    int tiles_H = (H_out + 3) / 4;  // Ceiling division
    int tiles_W = (W_out + 3) / 4;
    int total_tiles = tiles_H * tiles_W;
    
    if (tile_idx >= total_tiles) return;
    
    // Calculate tile coordinates
    int tile_h = tile_idx / tiles_W;
    int tile_w = tile_idx % tiles_W;
    
    // Shared memory for kernel transformation (assuming 3x3 kernel)
    extern __shared__ float shared_mem[];
    float* shared_weight = shared_mem;
    float* shared_input_tile = shared_mem + 36 * C_in;
    float* shared_output_tile = shared_mem + 36 * C_in + 36;
    
    // Load and transform kernel weights into shared memory
    if (threadIdx.x < C_in) {
        float kernel_data[9];
        for (int i = 0; i < 9; i++) {
            int weight_idx = channel_out_idx * C_in * 9 + threadIdx.x * 9 + i;
            kernel_data[i] = weight[weight_idx];
        }
        winograd_transform_kernel(kernel_data, &shared_weight[threadIdx.x * 36]);
    }
    __syncthreads();
    
    // Process input tile
    float input_tile[36] = {0.0f};
    float transformed_input[36] = {0.0f};
    float transformed_output[36] = {0.0f};
    float output_tile[16] = {0.0f};
    
    // Load input data for this tile
    int input_h_start = tile_h * 4 - 1;
    int input_w_start = tile_w * 4 - 1;
    
    for (int c = 0; c < C_in; c++) {
        // Reset input tile for each channel
        for (int i = 0; i < 36; i++) {
            input_tile[i] = 0.0f;
        }
        
        // Load input tile (with padding if needed)
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                int h = input_h_start + i;
                int w = input_w_start + j;
                
                if (h >= 0 && h < H_in && w >= 0 && w < W_in) {
                    int input_idx = batch_idx * C_in * H_in * W_in + 
                                   c * H_in * W_in + 
                                   h * W_in + w;
                    input_tile[i * 6 + j] = input[input_idx];
                }
            }
        }
        
        // Transform input tile
        winograd_transform_input(input_tile, transformed_input);
        
        // Element-wise multiply with transformed kernel
        for (int i = 0; i < 36; i++) {
            transformed_output[i] += transformed_input[i] * shared_weight[c * 36 + i];
        }
    }
    
    // Transform output back
    winograd_transform_output(transformed_output, output_tile);
    
    // Add bias and apply tanh activation
    float bias_val = bias[channel_out_idx];
    
    // Write output tile to global memory
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int out_h = tile_h * 4 + i;
            int out_w = tile_w * 4 + j;
            
            if (out_h < H_out && out_w < W_out) {
                int output_idx = batch_idx * C_out * H_out * W_out + 
                                channel_out_idx * H_out * W_out + 
                                out_h * W_out + out_w;
                float val = output_tile[i * 4 + j] + bias_val;
                output[output_idx] = tanhf(val);
            }
        }
    }
}

void winograd_conv_transpose_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    
    int C_out = weight.size(0);
    int H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    // Calculate number of tiles
    int tiles_H = (H_out + 3) / 4;
    int tiles_W = (W_out + 3) / 4;
    int total_tiles = tiles_H * tiles_W;
    
    // Kernel launch configuration
    dim3 grid(N, C_out, (total_tiles + 255) / 256);
    dim3 block(256);
    
    // Shared memory size: weights (36*C_in) + input_tile (36) + output_tile (16)
    size_t shared_mem_size = (36 * C_in + 36 + 16) * sizeof(float);
    
    winograd_conv_transpose_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out, H_in, W_in, H_out, W_out, kernel_size
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void winograd_conv_transpose_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("winograd_conv_transpose_forward", &winograd_conv_transpose_forward, "Winograd optimized conv transpose forward pass");
}
"""

# Compile the extension
winograd_ext = load_inline(
    name='winograd_ext',
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
    # Allocate output tensor
    N, C_in, H_in, W_in = x.size()
    kernel_size = conv_transpose_weight.size(2)  # Assuming square kernel
    C_out = conv_transpose_weight.size(0)
    
    H_out = (H_in - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + kernel_size + conv_transpose_output_padding[0]
    W_out = (W_in - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + kernel_size + conv_transpose_output_padding[1]
    
    output = torch.empty(N, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Call optimized Winograd convolution
    winograd_ext.winograd_conv_transpose_forward(
        x, conv_transpose_weight, conv_transpose_bias, output,
        kernel_size, conv_transpose_stride[0], conv_transpose_padding[0], conv_transpose_output_padding[0]
    )
    
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
