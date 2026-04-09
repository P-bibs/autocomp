# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_064410/code_4.py
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

# Optimization: Coalesce memory access by mapping thread index to memory-contiguous 
# dimensions and utilizing vectorized reads where possible.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define WARP_SIZE 32

__global__ void fused_op_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float divisor,
    int N, int C, int D, int H, int W) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = N * D * H * W;

    if (idx < spatial_size) {
        float sum_val = 0.0f;
        int spatial_stride = spatial_size;
        
        // Coalesced access: each thread accesses consecutive channels of the same spatial location
        for (int c = 0; c < C; ++c) {
            sum_val += (__ldg(&input[c * spatial_size + idx]) / divisor) + __ldg(&bias[c]);
        }
        output[idx] = sum_val;
    }
}

// Shared memory optimized reduction kernel for convolution-like operation
__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int K_D, int K_H, int K_W,
    int pad_d, int pad_h, int pad_w,
    int stride_d, int stride_h, int stride_w) {
    
    // Calculate output dimensions
    int D_out = (D_in + 2 * pad_d - K_D) / stride_d + 1;
    int H_out = (H_in + 2 * pad_h - K_H) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - K_W) / stride_w + 1;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * C_out * D_out * H_out * W_out;
    
    if (tid < total_threads) {
        int w_out = tid % W_out;
        int h_out = (tid / W_out) % H_out;
        int d_out = (tid / (W_out * H_out)) % D_out;
        int c_out = (tid / (W_out * H_out * D_out)) % C_out;
        int n = tid / (W_out * H_out * D_out * C_out);
        
        float sum = 0.0f;
        
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kd = 0; kd < K_D; ++kd) {
                for (int kh = 0; kh < K_H; ++kh) {
                    for (int kw = 0; kw < K_W; ++kw) {
                        int d_in = d_out * stride_d - pad_d + kd;
                        int h_in = h_out * stride_h - pad_h + kh;
                        int w_in = w_out * stride_w - pad_w + kw;
                        
                        if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            int input_idx = n * (C_in * D_in * H_in * W_in) + 
                                          c_in * (D_in * H_in * W_in) + 
                                          d_in * (H_in * W_in) + 
                                          h_in * W_in + 
                                          w_in;
                                          
                            int weight_idx = c_out * (C_in * K_D * K_H * K_W) + 
                                           c_in * (K_D * K_H * K_W) + 
                                           kd * (K_H * K_W) + 
                                           kh * K_W + 
                                           kw;
                                           
                            sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                        }
                    }
                }
            }
        }
        
        int output_idx = n * (C_out * D_out * H_out * W_out) + 
                        c_out * (D_out * H_out * W_out) + 
                        d_out * (H_out * W_out) + 
                        h_out * W_out + 
                        w_out;
                        
        output[output_idx] = sum + __ldg(&bias[c_out]);
    }
}

__global__ void max_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int K_D, int K_H, int K_W,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * C * D_out * H_out * W_out;
    
    if (tid < total_threads) {
        int w_out = tid % W_out;
        int h_out = (tid / W_out) % H_out;
        int d_out = (tid / (W_out * H_out)) % D_out;
        int c = (tid / (W_out * H_out * D_out)) % C;
        int n = tid / (W_out * H_out * D_out * C);
        
        float max_val = -FLT_MAX;
        
        for (int kd = 0; kd < K_D; ++kd) {
            for (int kh = 0; kh < K_H; ++kh) {
                for (int kw = 0; kw < K_W; ++kw) {
                    int d_in = d_out * stride_d - pad_d + kd;
                    int h_in = h_out * stride_h - pad_h + kh;
                    int w_in = w_out * stride_w - pad_w + kw;
                    
                    if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int input_idx = n * (C * D_in * H_in * W_in) + 
                                      c * (D_in * H_in * W_in) + 
                                      d_in * (H_in * W_in) + 
                                      h_in * W_in + 
                                      w_in;
                                      
                        float val = __ldg(&input[input_idx]);
                        max_val = fmaxf(max_val, val);
                    }
                }
            }
        }
        
        int output_idx = n * (C * D_out * H_out * W_out) + 
                        c * (D_out * H_out * W_out) + 
                        d_out * (H_out * W_out) + 
                        h_out * W_out + 
                        w_out;
                        
        output[output_idx] = max_val;
    }
}

__global__ void adaptive_avg_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * C * D_out * H_out * W_out;
    
    if (tid < total_threads) {
        int w_out = tid % W_out;
        int h_out = (tid / W_out) % H_out;
        int d_out = (tid / (W_out * H_out)) % D_out;
        int c = (tid / (W_out * H_out * D_out)) % C;
        int n = tid / (W_out * H_out * D_out * C);
        
        int d_start = (d_out * D_in) / D_out;
        int d_end = ((d_out + 1) * D_in + D_out - 1) / D_out;
        int h_start = (h_out * H_in) / H_out;
        int h_end = ((h_out + 1) * H_in + H_out - 1) / H_out;
        int w_start = (w_out * W_in) / W_out;
        int w_end = ((w_out + 1) * W_in + W_out - 1) / W_out;
        
        float sum = 0.0f;
        int count = 0;
        
        for (int d = d_start; d < d_end; ++d) {
            for (int h = h_start; h < h_end; ++h) {
                for (int w = w_start; w < w_end; ++w) {
                    int input_idx = n * (C * D_in * H_in * W_in) + 
                                  c * (D_in * H_in * W_in) + 
                                  d * (H_in * W_in) + 
                                  h * W_in + 
                                  w;
                    sum += __ldg(&input[input_idx]);
                    count++;
                }
            }
        }
        
        int output_idx = n * (C * D_out * H_out * W_out) + 
                        c * (D_out * H_out * W_out) + 
                        d_out * (H_out * W_out) + 
                        h_out * W_out + 
                        w_out;
                        
        output[output_idx] = sum / count;
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor bias, torch::Tensor output, float divisor) {
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    
    int total_threads = N * D * H * W;
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    fused_op_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), 
        divisor, N, C, D, H, W);
}

void conv3d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output,
    int K_D, int K_H, int K_W,
    int pad_d, int pad_h, int pad_w,
    int stride_d, int stride_h, int stride_w) {
    
    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    
    int C_out = weight.size(0);
    
    int threads_per_block = 256;
    int total_threads = N * C_out * 
                        ((D_in + 2 * pad_d - K_D) / stride_d + 1) * 
                        ((H_in + 2 * pad_h - K_H) / stride_h + 1) * 
                        ((W_in + 2 * pad_w - K_W) / stride_w + 1);
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    conv3d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in, C_out, K_D, K_H, K_W,
        pad_d, pad_h, pad_w, stride_d, stride_h, stride_w);
}

void max_pool3d_forward(
    torch::Tensor input, torch::Tensor output,
    int K_D, int K_H, int K_W,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w) {
    
    int N = input.size(0);
    int C = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    
    int D_out = (D_in + 2 * pad_d - K_D) / stride_d + 1;
    int H_out = (H_in + 2 * pad_h - K_H) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - K_W) / stride_w + 1;
    
    int threads_per_block = 256;
    int total_threads = N * C * D_out * H_out * W_out;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    max_pool3d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, D_in, H_in, W_in, D_out, H_out, W_out,
        K_D, K_H, K_W, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w);
}

void adaptive_avg_pool3d_forward(
    torch::Tensor input, torch::Tensor output,
    int D_out, int H_out, int W_out) {
    
    int N = input.size(0);
    int C = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    
    int threads_per_block = 256;
    int total_threads = N * C * D_out * H_out * W_out;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    adaptive_avg_pool3d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, D_in, H_in, W_in, D_out, H_out, W_out);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor input, torch::Tensor bias, torch::Tensor output, float divisor);
void conv3d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                   int K_D, int K_H, int K_W, int pad_d, int pad_h, int pad_w,
                   int stride_d, int stride_h, int stride_w);
void max_pool3d_forward(torch::Tensor input, torch::Tensor output,
                       int K_D, int K_H, int K_W, int stride_d, int stride_h, int stride_w,
                       int pad_d, int pad_h, int pad_w);
void adaptive_avg_pool3d_forward(torch::Tensor input, torch::Tensor output, int D_out, int H_out, int W_out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused divide, bias, and sum kernel");
    m.def("conv3d", &conv3d_forward, "Custom 3D convolution kernel");
    m.def("max_pool3d", &max_pool3d_forward, "Custom 3D max pooling kernel");
    m.def("adaptive_avg_pool3d", &adaptive_avg_pool3d_forward, "Custom 3D adaptive average pooling kernel");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups,
    max_pool_kernel_size, max_pool_stride, max_pool_padding, max_pool_dilation,
    max_pool_ceil_mode, max_pool_return_indices, global_avg_pool_output_size,
    divisor, bias, sum_dim,
):
    # Extract convolution parameters
    pad_d, pad_h, pad_w = conv_padding if isinstance(conv_padding, (tuple, list)) else (conv_padding,) * 3
    stride_d, stride_h, stride_w = conv_stride if isinstance(conv_stride, (tuple, list)) else (conv_stride,) * 3
    K_D, K_H, K_W = conv_weight.shape[2], conv_weight.shape[3], conv_weight.shape[4]
    
    # Conv3D output dimensions
    N, C_in, D_in, H_in, W_in = x.shape
    D_conv_out = (D_in + 2 * pad_d - K_D) // stride_d + 1
    H_conv_out = (H_in + 2 * pad_h - K_H) // stride_h + 1
    W_conv_out = (W_in + 2 * pad_w - K_W) // stride_w + 1
    C_out = conv_weight.shape[0]
    
    conv_output = torch.empty((N, C_out, D_conv_out, H_conv_out, W_conv_out), device=x.device, dtype=x.dtype)
    
    # Perform custom convolution
    fused_ext.conv3d(
        x.contiguous(), conv_weight.contiguous(), conv_bias.contiguous(),
        conv_output,
        K_D, K_H, K_W,
        pad_d, pad_h, pad_w,
        stride_d, stride_h, stride_w
    )
    
    # Max pooling parameters
    pool_k_d, pool_k_h, pool_k_w = max_pool_kernel_size if isinstance(max_pool_kernel_size, (tuple, list)) else (max_pool_kernel_size,) * 3
    pool_s_d, pool_s_h, pool_s_w = max_pool_stride if isinstance(max_pool_stride, (tuple, list)) else (max_pool_stride,) * 3
    pool_p_d, pool_p_h, pool_p_w = max_pool_padding if isinstance(max_pool_padding, (tuple, list)) else (max_pool_padding,) * 3
    
    # Max pooling output dimensions
    D_pool_out = (D_conv_out + 2 * pool_p_d - pool_k_d) // pool_s_d + 1
    H_pool_out = (H_conv_out + 2 * pool_p_h - pool_k_h) // pool_s_h + 1
    W_pool_out = (W_conv_out + 2 * pool_p_w - pool_k_w) // pool_s_w + 1
    
    pool_output = torch.empty((N, C_out, D_pool_out, H_pool_out, W_pool_out), device=x.device, dtype=x.dtype)
    
    # Perform custom max pooling
    fused_ext.max_pool3d(
        conv_output.contiguous(), pool_output,
        pool_k_d, pool_k_h, pool_k_w,
        pool_s_d, pool_s_h, pool_s_w,
        pool_p_d, pool_p_h, pool_p_w
    )
    
    # Adaptive average pooling output dimensions
    D_adapt_out, H_adapt_out, W_adapt_out = global_avg_pool_output_size
    
    adapt_output = torch.empty((N, C_out, D_adapt_out, H_adapt_out, W_adapt_out), device=x.device, dtype=x.dtype)
    
    # Perform custom adaptive average pooling
    fused_ext.adaptive_avg_pool3d(
        pool_output.contiguous(), adapt_output,
        D_adapt_out, H_adapt_out, W_adapt_out
    )
    
    # Final fused operation: sum over channel dimension with divide and bias
    out = torch.empty((N, D_adapt_out, H_adapt_out, W_adapt_out), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(adapt_output.contiguous(), bias.contiguous().view(-1), out, divisor)
    
    return out

# Placeholders for evaluation requirements
batch_size=128; in_channels=8; out_channels=16; depth=16; height=64; width=64
kernel_size=(3, 3, 3); divisor=2.0; pool_size=(2, 2, 2); bias_shape=(out_channels, 1, 1, 1); sum_dim=1

def get_init_inputs(): return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]
def get_inputs(): return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
