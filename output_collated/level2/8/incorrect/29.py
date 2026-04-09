# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_060810/code_3.py
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

# CUDA kernel for fused operations: Conv3D + Division + MaxPool3D + AdaptiveAvgPool3D + Bias Add + Sum
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

// Function to compute 3D index from 5D tensor (N, C, D, H, W)
__device__ inline int get_index_5d(int n, int c, int d, int h, int w, int C, int D, int H, int W) {
    return ((n * C + c) * D + d) * H * W + h * W + w;
}

__device__ inline int get_index_4d(int n, int d, int h, int w, int D, int H, int W) {
    return (n * D + d) * H * W + h * W + w;
}

// Custom Conv3D + MaxPool3D + AdaptiveAvgPool3D fused kernel
__global__ void fused_op_kernel(
    const float* __restrict__ input,         // [N, Ci, Di, Hi, Wi]
    const float* __restrict__ weight,        // [Co, Ci, Kd, Kh, Kw]
    const float* __restrict__ conv_bias,     // [Co]
    float divisor,
    const float* __restrict__ bias,          // [Co]
    int N, int Ci, int Di, int Hi, int Wi,
    int Co,
    int Kd, int Kh, int Kw,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int max_pool_kd, int max_pool_kh, int max_pool_kw,
    int max_pool_stride_d, int max_pool_stride_h, int max_pool_stride_w,
    int max_pool_pad_d, int max_pool_pad_h, int max_pool_pad_w,
    int global_D, int global_H, int global_W,
    int sum_dim,
    float* __restrict__ output
) {
    // Each block handles one output element in the final reduced tensor [N, D', H', W']
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = N * global_D * global_H * global_W;

    if (tid >= total_output_elements) return;

    int n = tid / (global_D * global_H * global_W);
    int d_idx = (tid / (global_H * global_W)) % global_D;
    int h_idx = (tid / global_W) % global_H;
    int w_idx = tid % global_W;

    // Compute adaptive pooling source region
    int src_d_start = (d_idx * Di) / global_D;
    int src_d_end = ((d_idx + 1) * Di + global_D - 1) / global_D;
    int src_h_start = (h_idx * Hi) / global_H;
    int src_h_end = ((h_idx + 1) * Hi + global_H - 1) / global_H;
    int src_w_start = (w_idx * Wi) / global_W;
    int src_w_end = ((w_idx + 1) * Wi + global_W - 1) / global_W;

    float sum_val = 0.0f;

    // Loop through all output channels
    for (int co = 0; co < Co; ++co) {
        float channel_sum = 0.0f;
        int pool_count = 0;

        // Loop through reduced spatial dimensions after adaptive pool
        for (int pd = src_d_start; pd < src_d_end; ++pd) {
            for (int ph = src_h_start; ph < src_h_end; ++ph) {
                for (int pw = src_w_start; pw < src_w_end; ++pw) {
                    float max_val = -FLT_MAX;

                    // Perform max pooling around (pd, ph, pw)
                    bool has_valid = false;
                    for (int md = 0; md < max_pool_kd; ++md) {
                        for (int mh = 0; mh < max_pool_kh; ++mh) {
                            for (int mw = 0; mw < max_pool_kw; ++mw) {
                                int in_d = pd * max_pool_stride_d + md - max_pool_pad_d;
                                int in_h = ph * max_pool_stride_h + mh - max_pool_pad_h;
                                int in_w = pw * max_pool_stride_w + mw - max_pool_pad_w;

                                if (in_d < 0 || in_d >= Di || in_h < 0 || in_h >= Hi || in_w < 0 || in_w >= Wi)
                                    continue;

                                // Compute conv3d at this location for channel co
                                float conv_val = 0.0f;
                                for (int ci = 0; ci < Ci; ++ci) {
                                    for (int kd = 0; kd < Kd; ++kd) {
                                        for (int kh = 0; kh < Kh; ++kh) {
                                            for (int kw = 0; kw < Kw; ++kw) {
                                                int in_d_conv = in_d - kd + pad_d;
                                                int in_h_conv = in_h - kh + pad_h;
                                                int in_w_conv = in_w - kw + pad_w;

                                                if (in_d_conv % stride_d != 0 || in_h_conv % stride_h != 0 || in_w_conv % stride_w != 0)
                                                    continue;

                                                in_d_conv /= stride_d;
                                                in_h_conv /= stride_h;
                                                in_w_conv /= stride_w;

                                                if (in_d_conv < 0 || in_d_conv >= (Di + 2*pad_d - Kd)/stride_d + 1 ||
                                                    in_h_conv < 0 || in_h_conv >= (Hi + 2*pad_h - Kh)/stride_h + 1 ||
                                                    in_w_conv < 0 || in_w_conv >= (Wi + 2*pad_w - Kw)/stride_w + 1)
                                                    continue;

                                                int inp_idx = get_index_5d(n, ci, in_d_conv, in_h_conv, in_w_conv, Ci, Di, Hi, Wi);
                                                int wgt_idx = get_index_5d(co, ci, kd, kh, kw, Ci, Kd, Kh, Kw);
                                                conv_val += input[inp_idx] * weight[wgt_idx];
                                            }
                                        }
                                    }
                                }
                                conv_val += conv_bias[co];
                                conv_val /= divisor;

                                if (!has_valid || conv_val > max_val) {
                                    max_val = conv_val;
                                    has_valid = true;
                                }
                            }
                        }
                    }

                    if (has_valid) {
                        channel_sum += max_val;
                        pool_count++;
                    }
                }
            }
        }

        // Average over the adaptive pooling region
        float avg_val = (pool_count > 0) ? channel_sum / pool_count : 0.0f;
        avg_val += bias[co]; // Add per-channel bias

        if (sum_dim == 1) {
            sum_val += avg_val;
        }
    }

    // Write to output
    int out_idx = get_index_4d(n, d_idx, h_idx, w_idx, global_D, global_H, global_W);
    output[out_idx] = sum_val;
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    float divisor,
    torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int max_pool_kd, int max_pool_kh, int max_pool_kw,
    int max_pool_stride_d, int max_pool_stride_h, int max_pool_stride_w,
    int max_pool_pad_d, int max_pool_pad_h, int max_pool_pad_w,
    int global_D, int global_H, int global_W,
    int sum_dim,
    torch::Tensor output
) {
    int N = input.size(0);
    int threads = 256;
    int total_elements = N * global_D * global_H * global_W;
    int blocks = (total_elements + threads - 1) / threads;

    fused_op_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        divisor,
        bias.data_ptr<float>(),
        N, (int)input.size(1), (int)input.size(2), (int)input.size(3), (int)input.size(4),
        (int)weight.size(0),
        (int)weight.size(2), (int)weight.size(3), (int)weight.size(4),
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        max_pool_kd, max_pool_kh, max_pool_kw,
        max_pool_stride_d, max_pool_stride_h, max_pool_stride_w,
        max_pool_pad_d, max_pool_pad_h, max_pool_pad_w,
        global_D, global_H, global_W,
        sum_dim,
        output.data_ptr<float>()
    );
}
"""

# C++ interface bindings
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    float divisor,
    torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int max_pool_kd, int max_pool_kh, int max_pool_kw,
    int max_pool_stride_d, int max_pool_stride_h, int max_pool_stride_w,
    int max_pool_pad_d, int max_pool_pad_h, int max_pool_pad_w,
    int global_D, int global_H, int global_W,
    int sum_dim,
    torch::Tensor output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv3D + Div + MaxPool3D + AdaptiveAvgPool3D + Bias + Sum");
}
"""

# Compile extension
fused_ext = load_inline(
    name='fused_op_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Model parameters
batch_size = 128
in_channels = 8
out_channels = 16
depth = height = width = 16
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

# Convolution parameters
conv_stride = (1, 1, 1)
conv_padding = (1, 1, 1)
conv_dilation = (1, 1, 1)
conv_groups = 1

# Max pooling parameters
max_pool_kernel_size = pool_size
max_pool_stride = pool_size
max_pool_padding = (0, 0, 0)
max_pool_dilation = (1, 1, 1)
max_pool_ceil_mode = False
max_pool_return_indices = False

# Global average pooling output size
global_avg_pool_output_size = (4, 4, 4)

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
    # Output shape after global avg pooling: [N, Co, D', H', W']
    N, Co, D_out, H_out, W_out = x.size(0), conv_weight.size(0), *global_avg_pool_output_size
    output = torch.zeros((N, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(
        x, conv_weight, conv_bias, divisor, bias.view(-1),
        conv_stride[0], conv_stride[1], conv_stride[2],
        conv_padding[0], conv_padding[1], conv_padding[2],
        max_pool_kernel_size[0], max_pool_kernel_size[1], max_pool_kernel_size[2],
        max_pool_stride[0], max_pool_stride[1], max_pool_stride[2],
        max_pool_padding[0], max_pool_padding[1], max_pool_padding[2],
        global_avg_pool_output_size[0], global_avg_pool_output_size[1], global_avg_pool_output_size[2],
        sum_dim, output
    )
    
    return output

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
