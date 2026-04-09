# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_064410/code_15.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for both Conv3D and the fused post-convolution operations
conv_and_fused_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Conv3D kernel
__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int K_d, int K_h, int K_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * C_out * D_out * H_out * W_out;
    
    if (idx >= total_threads) return;

    int w_out = idx % W_out;
    idx /= W_out;
    int h_out = idx % H_out;
    idx /= H_out;
    int d_out = idx % D_out;
    idx /= D_out;
    int c_out = idx % C_out;
    int n = idx / C_out;

    float sum = 0.0f;
    
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kd = 0; kd < K_d; ++kd) {
            int d_in = d_out * stride_d - padding_d + kd * dilation_d;
            if (d_in < 0 || d_in >= D_in) continue;
            
            for (int kh = 0; kh < K_h; ++kh) {
                int h_in = h_out * stride_h - padding_h + kh * dilation_h;
                if (h_in < 0 || h_in >= H_in) continue;
                
                for (int kw = 0; kw < K_w; ++kw) {
                    int w_in = w_out * stride_w - padding_w + kw * dilation_w;
                    if (w_in < 0 || w_in >= W_in) continue;
                    
                    int input_idx = ((n * C_in + c_in) * D_in + d_in) * H_in + h_in;
                    input_idx = input_idx * W_in + w_in;
                    
                    int weight_idx = ((c_out * C_in + c_in) * K_d + kd) * K_h + kh;
                    weight_idx = weight_idx * K_w + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    sum += bias[c_out];
    
    int output_idx = ((((n * C_out + c_out) * D_out + d_out) * H_out + h_out) * W_out + w_out);
    output[output_idx] = sum;
}

// Fused post-convolution kernel
__global__ void fused_post_conv_kernel(
    const float* __restrict__ x,
    float* __restrict__       sum_max,
    const int N, const int C,
    const int D, const int H, const int W,
    const int D_out, const int H_out, const int W_out,
    const int pool_kernel_d, const int pool_kernel_h, const int pool_kernel_w,
    const int pool_stride_d, const int pool_stride_h, const int pool_stride_w,
    const int pool_padding_d, const int pool_padding_h, const int pool_padding_w,
    const int pool_dilation_d, const int pool_dilation_h, const int pool_dilation_w,
    const float divisor)
{
    const int total_windows = N * C * D_out * H_out * W_out;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_windows) return;

    int rest = tid;
    const int windows_per_batch = C * D_out * H_out * W_out;
    const int batch = rest / windows_per_batch;
    rest %= windows_per_batch;

    const int windows_per_channel = D_out * H_out * W_out;
    const int ch = rest / windows_per_channel;
    rest %= windows_per_channel;

    const int d_out = rest / (H_out * W_out);
    rest %= (H_out * W_out);
    const int h_out = rest / W_out;
    const int w_out = rest % W_out;

    float max_val = -1e38f;

    for (int kd = 0; kd < pool_kernel_d; ++kd) {
        int d_in = d_out * pool_stride_d + kd * pool_dilation_d - pool_padding_d;
        if (d_in < 0 || d_in >= D) continue;
        for (int kh = 0; kh < pool_kernel_h; ++kh) {
            int h_in = h_out * pool_stride_h + kh * pool_dilation_h - pool_padding_h;
            if (h_in < 0 || h_in >= H) continue;
            for (int kw = 0; kw < pool_kernel_w; ++kw) {
                int w_in = w_out * pool_stride_w + kw * pool_dilation_w - pool_padding_w;
                if (w_in < 0 || w_in >= W) continue;

                int idx = ((batch * C + ch) * D + d_in) * H + h_in;
                idx = idx * W + w_in;

                float v = x[idx];
                if (v > max_val) max_val = v;
            }
        }
    }

    max_val = max_val / divisor;
    atomicAdd(&sum_max[batch], max_val);
}

void launch_conv3d(
    const float* input, const float* weight, const float* bias,
    float* output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int K_d, int K_h, int K_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int D_out, int H_out, int W_out) {
    
    int total_threads = N * C_out * D_out * H_out * W_out;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;
    
    conv3d_kernel<<<blocks, threads>>>(
        input, weight, bias, output,
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        K_d, K_h, K_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w
    );
}

void launch_fused_post_conv(
    const float* x,
    float* sum_max,
    int N, int C, int D, int H, int W,
    int pool_kernel_d, int pool_kernel_h, int pool_kernel_w,
    int pool_stride_d, int pool_stride_h, int pool_stride_w,
    int pool_padding_d, int pool_padding_h, int pool_padding_w,
    int pool_dilation_d, int pool_dilation_h, int pool_dilation_w,
    float divisor) {
    
    const int D_out = (D + 2 * pool_padding_d - pool_dilation_d * (pool_kernel_d - 1) - 1) / pool_stride_d + 1;
    const int H_out = (H + 2 * pool_padding_h - pool_dilation_h * (pool_kernel_h - 1) - 1) / pool_stride_h + 1;
    const int W_out = (W + 2 * pool_padding_w - pool_dilation_w * (pool_kernel_w - 1) - 1) / pool_stride_w + 1;

    const int total_windows = N * C * D_out * H_out * W_out;
    const int threads = 256;
    const int blocks  = (total_windows + threads - 1) / threads;

    fused_post_conv_kernel<<<blocks, threads>>>(
        x, sum_max,
        N, C, D, H, W,
        D_out, H_out, W_out,
        pool_kernel_d, pool_kernel_h, pool_kernel_w,
        pool_stride_d, pool_stride_h, pool_stride_w,
        pool_padding_d, pool_padding_h, pool_padding_w,
        pool_dilation_d, pool_dilation_h, pool_dilation_w,
        divisor);
}
"""

# C++ binding
conv_and_fused_cpp_source = r"""
#include <torch/extension.h>

void launch_conv3d(
    const float* input, const float* weight, const float* bias,
    float* output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int K_d, int K_h, int K_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int D_out, int H_out, int W_out);

void launch_fused_post_conv(
    const float* x,
    float* sum_max,
    int N, int C, int D, int H, int W,
    int pool_kernel_d, int pool_kernel_h, int pool_kernel_w,
    int pool_stride_d, int pool_stride_h, int pool_stride_w,
    int pool_padding_d, int pool_padding_h, int pool_padding_w,
    int pool_dilation_d, int pool_dilation_h, int pool_dilation_w,
    float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_conv3d", &launch_conv3d, "Launch custom Conv3D kernel");
    m.def("launch_fused_post_conv", &launch_fused_post_conv, "Launch fused post-conv kernel");
}
"""

# Compile the extension
custom_ext = load_inline(
    name="custom_ext",
    cpp_sources=conv_and_fused_cpp_source,
    cuda_sources=conv_and_fused_cuda_source,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)

def _pool_out_size(in_size, kernel, stride, padding, dilation):
    return (in_size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    max_pool_kernel_size,
    max_pool_stride,
    max_pool_padding,
    max_pool_dilation,
    max_pool_ceil_mode,
    max_pool_return_indices,
    global_avg_pool_output_size,
    divisor,
    bias,
    sum_dim,
):
    # Get input dimensions
    N, C_in, D_in, H_in, W_in = x.shape
    C_out, _, K_d, K_h, K_w = conv_weight.shape
    
    # Calculate output dimensions for conv3d
    stride_d, stride_h, stride_w = conv_stride
    padding_d, padding_h, padding_w = conv_padding
    dilation_d, dilation_h, dilation_w = conv_dilation
    
    D_out = (D_in + 2 * padding_d - dilation_d * (K_d - 1) - 1) // stride_d + 1
    H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) // stride_w + 1
    
    # Allocate output tensor for conv3d
    conv_output = torch.empty(N, C_out, D_out, H_out, W_out, dtype=torch.float32, device=x.device)
    
    # Launch custom Conv3D kernel
    custom_ext.launch_conv3d(
        x.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        conv_output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, K_d, K_h, K_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w,
        D_out, H_out, W_out
    )
    
    # Prepare pool parameters
    kd, kh, kw = max_pool_kernel_size
    sd, sh, sw = (max_pool_stride if max_pool_stride is not None else max_pool_kernel_size)
    pd, ph, pw = max_pool_padding
    dd, dh, dw = max_pool_dilation

    # Allocate accumulator for the fused kernel
    sum_max = torch.zeros(N, dtype=torch.float32, device=x.device)

    # Launch fused kernel
    custom_ext.launch_fused_post_conv(
        conv_output.data_ptr<float>(),
        sum_max.data_ptr<float>(),
        N, C_out, D_out, H_out, W_out,
        kd, kh, kw,
        sd, sh, sw,
        pd, ph, pw,
        dd, dh, dw,
        float(divisor)
    )

    # Complete the computation on the GPU
    # Number of pooling windows per channel
    D_pool_out = _pool_out_size(D_out, kd, sd, pd, dd)
    H_pool_out = _pool_out_size(H_out, kh, sh, ph, dh)
    W_pool_out = _pool_out_size(W_out, kw, sw, pw, dw)
    windows_per_channel = D_pool_out * H_pool_out * W_pool_out
    total_windows_per_batch = C_out * windows_per_channel

    # Sum of bias across channels
    bias_sum = bias.sum().item()

    # Final result: average of scaled max values + summed bias
    final = (sum_max / total_windows_per_batch + bias_sum).view(N, 1, 1, 1)

    return final

batch_size   = 128  
in_channels  = 8            
out_channels = 16  
depth = 16; height = width = 64 
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
