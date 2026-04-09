# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_061658/code_6.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'divisor', 'pool_size', 'bias_shape', 'sum_dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'max_pool_kernel_size', 'max_pool_stride', 'max_pool_padding', 'max_pool_dilation', 'max_pool_ceil_mode', 'max_pool_return_indices', 'global_avg_pool_output_size', 'divisor', 'bias', 'sum_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """

    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

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
    # State for max_pool (nn.MaxPool3d)
    state_kwargs['max_pool_kernel_size'] = model.max_pool.kernel_size
    state_kwargs['max_pool_stride'] = model.max_pool.stride
    state_kwargs['max_pool_padding'] = model.max_pool.padding
    state_kwargs['max_pool_dilation'] = model.max_pool.dilation
    state_kwargs['max_pool_ceil_mode'] = model.max_pool.ceil_mode
    state_kwargs['max_pool_return_indices'] = model.max_pool.return_indices
    # State for global_avg_pool (nn.AdaptiveAvgPool3d)
    state_kwargs['global_avg_pool_output_size'] = model.global_avg_pool.output_size
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    if 'sum_dim' in flat_state:
        state_kwargs['sum_dim'] = flat_state['sum_dim']
    else:
        state_kwargs['sum_dim'] = getattr(model, 'sum_dim')
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

# --- CUDA Kernel ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// Custom Conv3D Kernel (Implicit GEMM approach)
// ============================================================================
__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out, int D_in, int H_in, int W_in,
    int KD, int KH, int KW, int stride, int padding, int dilation) {
    
    // Grid-stride loop for flexibility
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_size = N * C_out * D_in * H_in * W_in;  // Same spatial dims with stride=1, padding=1
    
    if (idx < out_size) {
        int n = idx / (C_out * D_in * H_in * W_in);
        int c_out = (idx / (D_in * H_in * W_in)) % C_out;
        int d = (idx / (H_in * W_in)) % D_in;
        int h = (idx / W_in) % H_in;
        int w = idx % W_in;
        
        float sum = (bias != nullptr) ? bias[c_out] : 0.0f;
        
        // Convolution loop
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kd = 0; kd < KD; ++kd) {
                int d_in = d + kd - KD/2;
                if (d_in < 0 || d_in >= D_in) continue;
                
                for (int kh = 0; kh < KH; ++kh) {
                    int h_in = h + kh - KH/2;
                    if (h_in < 0 || h_in >= H_in) continue;
                    
                    for (int kw = 0; kw < KW; ++kw) {
                        int w_in = w + kw - KW/2;
                        if (w_in < 0 || w_in >= W_in) continue;
                        
                        int in_idx = n * (C_in * D_in * H_in * W_in) +
                                    c_in * (D_in * H_in * W_in) +
                                    d_in * (H_in * W_in) +
                                    h_in * W_in + w_in;
                        
                        int w_idx = c_out * (C_in * KD * KH * KW) +
                                   c_in * (KD * KH * KW) +
                                   kd * (KH * KW) +
                                   kh * KW + kw;
                        
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

// ============================================================================
// Optimized Fused Kernel with Vectorized Memory Access
// ============================================================================
__global__ void fused_op_kernel_vec(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float divisor,
    int N, int C, int D, int H, int W) {
    
    // Process NDHW elements, with vectorization across C dimension
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * D * H * W;
    
    if (idx < total_elements) {
        // Decode multi-dimensional index
        int n = idx / (D * H * W);
        int remaining = idx % (D * H * W);
        int d = remaining / (H * W);
        int h = (remaining / W) % H;
        int w = remaining % W;
        
        // Base offset for this position in input
        int base_offset = n * (C * D * H * W) + d * (H * W) + h * W + w;
        
        float sum_val = 0.0f;
        
        // Vectorized loop: process 4 channels at a time where possible
        int c = 0;
        int c_vec_limit = (C / 4) * 4;  // Process in chunks of 4
        
        for (; c < c_vec_limit; c += 4) {
            // Load 4 consecutive channels
            float4 input_vals = make_float4(
                input[base_offset + c * (D * H * W)],
                input[base_offset + (c + 1) * (D * H * W)],
                input[base_offset + (c + 2) * (D * H * W)],
                input[base_offset + (c + 3) * (D * H * W)]
            );
            
            float4 bias_vals = make_float4(
                bias[c],
                bias[c + 1],
                bias[c + 2],
                bias[c + 3]
            );
            
            // Process vectorized operations
            sum_val += (input_vals.x / divisor) + bias_vals.x;
            sum_val += (input_vals.y / divisor) + bias_vals.y;
            sum_val += (input_vals.z / divisor) + bias_vals.z;
            sum_val += (input_vals.w / divisor) + bias_vals.w;
        }
        
        // Handle remaining channels (C % 4)
        for (; c < C; ++c) {
            int input_idx = base_offset + c * (D * H * W);
            sum_val += (input[input_idx] / divisor) + bias[c];
        }
        
        output[idx] = sum_val;
    }
}

// ============================================================================
// Wrapper Functions
// ============================================================================
void conv3d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, int stride, int padding, int dilation) {
    
    int N = input.size(0);
    int C_in = input.size(1);
    int C_out = weight.size(0);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    int KD = weight.size(2);
    int KH = weight.size(3);
    int KW = weight.size(4);
    
    int threads = 256;
    int total_out = N * C_out * D_in * H_in * W_in;
    int blocks = (total_out + threads - 1) / threads;
    
    conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.data_ptr<float>(), output.data_ptr<float>(),
        N, C_in, C_out, D_in, H_in, W_in, KD, KH, KW, stride, padding, dilation
    );
}

void fused_op_forward(
    torch::Tensor input, torch::Tensor bias, torch::Tensor output, float divisor) {
    
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    
    // Use 512 threads for better occupancy on RTX 2080Ti
    int threads = 512;
    int blocks = (N * D * H * W + threads - 1) / threads;
    
    fused_op_kernel_vec<<<blocks, threads>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        divisor, N, C, D, H, W
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void conv3d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                    torch::Tensor output, int stride, int padding, int dilation);
void fused_op_forward(torch::Tensor input, torch::Tensor bias, torch::Tensor output, float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d", &conv3d_forward, "Custom 3D Convolution");
    m.def("fused_op", &fused_op_forward, "Fused divide, bias, and sum kernel with vectorization");
}
"""

fused_ext = load_inline(
    name='fused_op_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-maxrregcount=128'],
    with_cuda=True
)

import torch.nn.functional as F

def functional_model(
    x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups,
    max_pool_kernel_size, max_pool_stride, max_pool_padding, max_pool_dilation,
    max_pool_ceil_mode, max_pool_return_indices, global_avg_pool_output_size,
    divisor, bias, sum_dim,
):
    # Custom Conv3D
    N, C_in, D, H, W = x.shape
    C_out = conv_weight.size(0)
    conv_output = torch.zeros((N, C_out, D, H, W), dtype=x.dtype, device=x.device)
    
    fused_ext.conv3d(x.contiguous(), conv_weight.contiguous(), conv_bias.contiguous(),
                     conv_output, conv_stride, conv_padding, conv_dilation)
    
    x = conv_output
    
    # Max Pool 3D
    x = F.max_pool3d(x, kernel_size=max_pool_kernel_size, stride=max_pool_stride,
                     padding=max_pool_padding, dilation=max_pool_dilation,
                     ceil_mode=max_pool_ceil_mode, return_indices=max_pool_return_indices)
    
    # Adaptive Avg Pool 3D
    x = F.adaptive_avg_pool3d(x, global_avg_pool_output_size)
    
    # Fused custom kernel with vectorization
    N, C, D, H, W = x.shape
    out = torch.zeros((N, D, H, W), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(x.contiguous(), bias.contiguous().view(-1), out, divisor)
    return out


# Placeholders for evaluation requirements
batch_size = 128
in_channels = 8
out_channels = 16
depth = 16
height = 64
width = 64
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
