# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_055948/code_2.py
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
#include <cub/cub.cuh>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int K_D, int K_H, int K_W,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    int od = (D_in + 2 * pad_d - dilation_d * (K_D - 1) - 1) / stride_d + 1;
    int oh = (H_in + 2 * pad_h - dilation_h * (K_H - 1) - 1) / stride_h + 1;
    int ow = (W_in + 2 * pad_w - dilation_w * (K_W - 1) - 1) / stride_w + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * C_out * od * oh * ow;
    
    if (idx < total_threads) {
        int tmp = idx;
        int w_out = tmp % ow; tmp /= ow;
        int h_out = tmp % oh; tmp /= oh;
        int d_out = tmp % od; tmp /= od;
        int c_out = tmp % C_out; tmp /= C_out;
        int n = tmp;

        float sum = 0.0f;
        for (int kd = 0; kd < K_D; ++kd) {
            for (int kh = 0; kh < K_H; ++kh) {
                for (int kw = 0; kw < K_W; ++kw) {
                    for (int c_in = 0; c_in < C_in; ++c_in) {
                        int d_in = d_out * stride_d - pad_d + kd * dilation_d;
                        int h_in = h_out * stride_h - pad_h + kh * dilation_h;
                        int w_in = w_out * stride_w - pad_w + kw * dilation_w;

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
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        sum += bias[c_out];
        int output_idx = n * (C_out * od * oh * ow) +
                         c_out * (od * oh * ow) +
                         d_out * (oh * ow) +
                         h_out * ow +
                         w_out;
        output[output_idx] = sum;
    }
}

__global__ void maxpool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D_in, int H_in, int W_in,
    int K_D, int K_H, int K_W,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    int od = (D_in + 2 * pad_d - dilation_d * (K_D - 1) - 1) / stride_d + 1;
    int oh = (H_in + 2 * pad_h - dilation_h * (K_H - 1) - 1) / stride_h + 1;
    int ow = (W_in + 2 * pad_w - dilation_w * (K_W - 1) - 1) / stride_w + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * od * oh * ow) return;

    int tmp = idx;
    int w_out = tmp % ow; tmp /= ow;
    int h_out = tmp % oh; tmp /= oh;
    int d_out = tmp % od; tmp /= od;
    int c = tmp % C; tmp /= C;
    int n = tmp;

    float max_val = -FLT_MAX;
    for (int kd = 0; kd < K_D; ++kd) {
        for (int kh = 0; kh < K_H; ++kh) {
            for (int kw = 0; kw < K_W; ++kw) {
                int d_in = d_out * stride_d - pad_d + kd * dilation_d;
                int h_in = h_out * stride_h - pad_h + kh * dilation_h;
                int w_in = w_out * stride_w - pad_w + kw * dilation_w;

                if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int input_idx = n * (C * D_in * H_in * W_in) +
                                    c * (D_in * H_in * W_in) +
                                    d_in * (H_in * W_in) +
                                    h_in * W_in +
                                    w_in;
                    float val = input[input_idx];
                    if (val > max_val) max_val = val;
                }
            }
        }
    }
    output[idx] = max_val;
}

__global__ void adaptive_avg_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * D_out * H_out * W_out) return;

    int tmp = idx;
    int w_out = tmp % W_out; tmp /= W_out;
    int h_out = tmp % H_out; tmp /= H_out;
    int d_out = tmp % D_out; tmp /= D_out;
    int c = tmp % C; tmp /= C;
    int n = tmp;

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
                sum += input[input_idx];
                count++;
            }
        }
    }
    output[idx] = sum / count;
}

__global__ void fused_reduction_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float divisor,
    int N, int C, int D, int H, int W) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = D * H * W;
    
    if (idx < N * spatial_size) {
        int n = idx / spatial_size;
        int rem = idx % spatial_size;

        float sum_val = 0.0f;
        #pragma unroll 4
        for (int c = 0; c < C; ++c) {
            float b = bias[c]; 
            float val = input[n * (C * spatial_size) + c * spatial_size + rem];
            sum_val += (val / divisor) + b;
        }
        output[idx] = sum_val;
    }
}

void launch_conv3d(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w) {
    
    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    
    int C_out = weight.size(0);
    int K_D = weight.size(2);
    int K_H = weight.size(3);
    int K_W = weight.size(4);

    int threads = 256;
    int od = (D_in + 2 * pad_d - dilation_d * (K_D - 1) - 1) / stride_d + 1;
    int oh = (H_in + 2 * pad_h - dilation_h * (K_H - 1) - 1) / stride_h + 1;
    int ow = (W_in + 2 * pad_w - dilation_w * (K_W - 1) - 1) / stride_w + 1;
    int total_threads = N * C_out * od * oh * ow;
    int blocks = (total_threads + threads - 1) / threads;

    conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in, C_out, K_D, K_H, K_W,
        stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dilation_d, dilation_h, dilation_w);
}

void launch_maxpool3d(
    torch::Tensor input, torch::Tensor output,
    int K_D, int K_H, int K_W,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w) {
    
    int N = input.size(0);
    int C = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    int threads = 256;
    int od = (D_in + 2 * pad_d - dilation_d * (K_D - 1) - 1) / stride_d + 1;
    int oh = (H_in + 2 * pad_h - dilation_h * (K_H - 1) - 1) / stride_h + 1;
    int ow = (W_in + 2 * pad_w - dilation_w * (K_W - 1) - 1) / stride_w + 1;
    int total_threads = N * C * od * oh * ow;
    int blocks = (total_threads + threads - 1) / threads;

    maxpool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, D_in, H_in, W_in, K_D, K_H, K_W,
        stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dilation_d, dilation_h, dilation_w);
}

void launch_adaptive_avg_pool3d(
    torch::Tensor input, torch::Tensor output,
    int D_out, int H_out, int W_out) {
    
    int N = input.size(0);
    int C = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    int threads = 256;
    int total_threads = N * C * D_out * H_out * W_out;
    int blocks = (total_threads + threads - 1) / threads;

    adaptive_avg_pool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, D_in, H_in, W_in, D_out, H_out, W_out);
}

void launch_fused_reduction(
    torch::Tensor input, torch::Tensor bias, torch::Tensor output, float divisor) {
    
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    int threads = 256;
    int total_elements = N * D * H * W;
    int blocks = (total_elements + threads - 1) / threads;

    fused_reduction_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), 
        divisor, N, C, D, H, W);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_conv3d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                   int stride_d, int stride_h, int stride_w,
                   int pad_d, int pad_h, int pad_w,
                   int dilation_d, int dilation_h, int dilation_w);

void launch_maxpool3d(torch::Tensor input, torch::Tensor output,
                      int K_D, int K_H, int K_W,
                      int stride_d, int stride_h, int stride_w,
                      int pad_d, int pad_h, int pad_w,
                      int dilation_d, int dilation_h, int dilation_w);

void launch_adaptive_avg_pool3d(torch::Tensor input, torch::Tensor output,
                                int D_out, int H_out, int W_out);

void launch_fused_reduction(torch::Tensor input, torch::Tensor bias, torch::Tensor output, float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d", &launch_conv3d, "Custom Conv3D kernel");
    m.def("maxpool3d", &launch_maxpool3d, "Custom MaxPool3D kernel");
    m.def("adaptive_avg_pool3d", &launch_adaptive_avg_pool3d, "Custom Adaptive AvgPool3D kernel");
    m.def("fused_reduction", &launch_fused_reduction, "Fused reduction kernel with bias and division");
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
    # Conv3D
    stride_d, stride_h, stride_w = conv_stride
    pad_d, pad_h, pad_w = conv_padding
    dilation_d, dilation_h, dilation_w = conv_dilation
    
    N, C_in, D_in, H_in, W_in = x.shape
    C_out, _, K_D, K_H, K_W = conv_weight.shape
    
    od = (D_in + 2 * pad_d - dilation_d * (K_D - 1) - 1) // stride_d + 1
    oh = (H_in + 2 * pad_h - dilation_h * (K_H - 1) - 1) // stride_h + 1
    ow = (W_in + 2 * pad_w - dilation_w * (K_W - 1) - 1) // stride_w + 1
    
    x = x.contiguous()
    conv_weight = conv_weight.contiguous()
    conv_bias = conv_bias.contiguous()
    
    out_conv = torch.empty((N, C_out, od, oh, ow), device=x.device, dtype=x.dtype)
    fused_ext.conv3d(x, conv_weight, conv_bias, out_conv, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dilation_d, dilation_h, dilation_w)
    
    # MaxPool3D
    if isinstance(max_pool_kernel_size, int):
        K_D, K_H, K_W = max_pool_kernel_size, max_pool_kernel_size, max_pool_kernel_size
    else:
        K_D, K_H, K_W = max_pool_kernel_size
    
    if isinstance(max_pool_stride, int):
        stride_d, stride_h, stride_w = max_pool_stride, max_pool_stride, max_pool_stride
    else:
        stride_d, stride_h, stride_w = max_pool_stride
    
    if isinstance(max_pool_padding, int):
        pad_d, pad_h, pad_w = max_pool_padding, max_pool_padding, max_pool_padding
    else:
        pad_d, pad_h, pad_w = max_pool_padding
    
    if isinstance(max_pool_dilation, int):
        dilation_d, dilation_h, dilation_w = max_pool_dilation, max_pool_dilation, max_pool_dilation
    else:
        dilation_d, dilation_h, dilation_w = max_pool_dilation
    
    N2, C2, D2, H2, W2 = out_conv.shape
    od2 = (D2 + 2 * pad_d - dilation_d * (K_D - 1) - 1) // stride_d + 1
    oh2 = (H2 + 2 * pad_h - dilation_h * (K_H - 1) - 1) // stride_h + 1
    ow2 = (W2 + 2 * pad_w - dilation_w * (K_W - 1) - 1) // stride_w + 1
    
    out_conv = out_conv.contiguous()
    out_pool = torch.empty((N2, C2, od2, oh2, ow2), device=x.device, dtype=x.dtype)
    fused_ext.maxpool3d(out_conv, out_pool, K_D, K_H, K_W, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dilation_d, dilation_h, dilation_w)
    
    # Adaptive Avg Pool3D
    if isinstance(global_avg_pool_output_size, int):
        D_out, H_out, W_out = global_avg_pool_output_size, global_avg_pool_output_size, global_avg_pool_output_size
    else:
        D_out, H_out, W_out = global_avg_pool_output_size
    
    N3, C3, D3, H3, W3 = out_pool.shape
    out_pool = out_pool.contiguous()
    out_adaptive = torch.empty((N3, C3, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    fused_ext.adaptive_avg_pool3d(out_pool, out_adaptive, D_out, H_out, W_out)
    
    # Fused Reduction
    N4, C4, D4, H4, W4 = out_adaptive.shape
    out_adaptive = out_adaptive.contiguous()
    bias = bias.contiguous().view(-1)
    out_final = torch.empty((N4, D4, H4, W4), device=x.device, dtype=x.dtype)
    fused_ext.fused_reduction(out_adaptive, bias, out_final, divisor)
    
    return out_final

# Placeholders for evaluation requirements
batch_size=128; in_channels=8; out_channels=16; depth=16; height=64; width=64
kernel_size=(3, 3, 3); divisor=2.0; pool_size=(2, 2, 2); bias_shape=(out_channels, 1, 1, 1); sum_dim=1

def get_init_inputs(): return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]
def get_inputs(): return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
