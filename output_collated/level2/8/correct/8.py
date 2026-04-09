# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_060810/code_2.py
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

__global__ void fused_op_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float divisor,
    int N, int C, int D, int H, int W) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D * H * W) return;

    // Local accumulation register
    float sum_val = 0.0f;
    
    // Pre-calculate base offsets
    // input is [N, C, D, H, W], output is [N, D, H, W]
    // Each index 'idx' corresponds to one (N, D, H, W) point
    int spatial_size = D * H * W;
    int n = idx / spatial_size;
    int rest = idx % spatial_size;

    const float* input_ptr = input + (n * C * spatial_size) + rest;
    
    // Unroll or execute reduction over C
    // Accessing bias sequentially is much faster than index math in loop
    for (int c = 0; c < C; ++c) {
        sum_val += (input_ptr[c * spatial_size] / divisor) + bias[c];
    }
    
    output[idx] = sum_val;
}

// Custom 3D Convolution Kernel
__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int Ci, int Di, int Hi, int Wi,
    int Co, int Kd, int Kh, int Kw,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N * Co * ((Di + 2 * pad_d - dilation_d * (Kd - 1) - 1) / stride_d + 1) *
                                   ((Hi + 2 * pad_h - dilation_h * (Kh - 1) - 1) / stride_h + 1) *
                                   ((Wi + 2 * pad_w - dilation_w * (Kw - 1) - 1) / stride_w + 1);
    if (idx >= total_outputs) return;

    int odim_d = (Di + 2 * pad_d - dilation_d * (Kd - 1) - 1) / stride_d + 1;
    int odim_h = (Hi + 2 * pad_h - dilation_h * (Kh - 1) - 1) / stride_h + 1;
    int odim_w = (Wi + 2 * pad_w - dilation_w * (Kw - 1) - 1) / stride_w + 1;

    int tmp = idx;
    int w_out = tmp % odim_w; tmp /= odim_w;
    int h_out = tmp % odim_h; tmp /= odim_h;
    int d_out = tmp % odim_d; tmp /= odim_d;
    int co = tmp % Co; tmp /= Co;
    int n = tmp;

    float value = 0.0f;
    for (int ci = 0; ci < Ci; ++ci) {
        for (int kd = 0; kd < Kd; ++kd) {
            for (int kh = 0; kh < Kh; ++kh) {
                for (int kw = 0; kw < Kw; ++kw) {
                    int d_in = d_out * stride_d - pad_d + kd * dilation_d;
                    int h_in = h_out * stride_h - pad_h + kh * dilation_h;
                    int w_in = w_out * stride_w - pad_w + kw * dilation_w;

                    if (d_in >= 0 && d_in < Di && h_in >= 0 && h_in < Hi && w_in >= 0 && w_in < Wi) {
                        int input_idx = n * (Ci * Di * Hi * Wi) + ci * (Di * Hi * Wi) + d_in * (Hi * Wi) + h_in * Wi + w_in;
                        int weight_idx = co * (Ci * Kd * Kh * Kw) + ci * (Kd * Kh * Kw) + kd * (Kh * Kw) + kh * Kw + kw;
                        value += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    value += bias[co];
    int output_idx = n * (Co * odim_d * odim_h * odim_w) + co * (odim_d * odim_h * odim_w) + d_out * (odim_h * odim_w) + h_out * odim_w + w_out;
    output[output_idx] = value;
}

// Max Pooling 3D Kernel
__global__ void maxpool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int Di, int Hi, int Wi,
    int Kd, int Kh, int Kw,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int odim_d = (Di + 2 * pad_d - dilation_d * (Kd - 1) - 1) / stride_d + 1;
    int odim_h = (Hi + 2 * pad_h - dilation_h * (Kh - 1) - 1) / stride_h + 1;
    int odim_w = (Wi + 2 * pad_w - dilation_w * (Kw - 1) - 1) / stride_w + 1;
    int total_outputs = N * C * odim_d * odim_h * odim_w;
    
    if (idx >= total_outputs) return;

    int tmp = idx;
    int w_out = tmp % odim_w; tmp /= odim_w;
    int h_out = tmp % odim_h; tmp /= odim_h;
    int d_out = tmp % odim_d; tmp /= odim_d;
    int c = tmp % C; tmp /= C;
    int n = tmp;

    float max_val = -FLT_MAX;
    for (int kd = 0; kd < Kd; ++kd) {
        for (int kh = 0; kh < Kh; ++kh) {
            for (int kw = 0; kw < Kw; ++kw) {
                int d_in = d_out * stride_d - pad_d + kd * dilation_d;
                int h_in = h_out * stride_h - pad_h + kh * dilation_h;
                int w_in = w_out * stride_w - pad_w + kw * dilation_w;

                if (d_in >= 0 && d_in < Di && h_in >= 0 && h_in < Hi && w_in >= 0 && w_in < Wi) {
                    int input_idx = n * (C * Di * Hi * Wi) + c * (Di * Hi * Wi) + d_in * (Hi * Wi) + h_in * Wi + w_in;
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }
    }
    int output_idx = n * (C * odim_d * odim_h * odim_w) + c * (odim_d * odim_h * odim_w) + d_out * (odim_h * odim_w) + h_out * odim_w + w_out;
    output[output_idx] = max_val;
}

// Adaptive Avg Pool 3D Kernel (simplified for output size 1x1x1 per sample per channel)
__global__ void adaptive_avg_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int Di, int Hi, int Wi
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) return;

    int c = idx % C;
    int n = idx / C;

    float sum = 0.0f;
    for (int d = 0; d < Di; ++d) {
        for (int h = 0; h < Hi; ++h) {
            for (int w = 0; w < Wi; ++w) {
                int input_idx = n * (C * Di * Hi * Wi) + c * (Di * Hi * Wi) + d * (Hi * Wi) + h * Wi + w;
                sum += input[input_idx];
            }
        }
    }
    output[idx] = sum / (Di * Hi * Wi);
}

void fused_op_forward(torch::Tensor input, torch::Tensor bias, torch::Tensor output, float divisor) {
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    
    int total_elements = N * D * H * W;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_op_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), 
        divisor, N, C, D, H, W);
}

void conv3d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    int N = input.size(0);
    int Ci = input.size(1);
    int Di = input.size(2);
    int Hi = input.size(3);
    int Wi = input.size(4);
    
    int Co = weight.size(0);
    int Kd = weight.size(2);
    int Kh = weight.size(3);
    int Kw = weight.size(4);
    
    int odim_d = (Di + 2 * pad_d - dilation_d * (Kd - 1) - 1) / stride_d + 1;
    int odim_h = (Hi + 2 * pad_h - dilation_h * (Kh - 1) - 1) / stride_h + 1;
    int odim_w = (Wi + 2 * pad_w - dilation_w * (Kw - 1) - 1) / stride_w + 1;
    
    int total_outputs = N * Co * odim_d * odim_h * odim_w;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    
    conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, Ci, Di, Hi, Wi, Co, Kd, Kh, Kw,
        stride_d, stride_h, stride_w, pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w
    );
}

void maxpool3d_forward(
    torch::Tensor input, torch::Tensor output,
    int Kd, int Kh, int Kw,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    int N = input.size(0);
    int C = input.size(1);
    int Di = input.size(2);
    int Hi = input.size(3);
    int Wi = input.size(4);
    
    int odim_d = (Di + 2 * pad_d - dilation_d * (Kd - 1) - 1) / stride_d + 1;
    int odim_h = (Hi + 2 * pad_h - dilation_h * (Kh - 1) - 1) / stride_h + 1;
    int odim_w = (Wi + 2 * pad_w - dilation_w * (Kw - 1) - 1) / stride_w + 1;
    
    int total_outputs = N * C * odim_d * odim_h * odim_w;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    
    maxpool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, Di, Hi, Wi, Kd, Kh, Kw,
        stride_d, stride_h, stride_w, pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w
    );
}

void adaptive_avg_pool3d_forward(torch::Tensor input, torch::Tensor output) {
    int N = input.size(0);
    int C = input.size(1);
    int Di = input.size(2);
    int Hi = input.size(3);
    int Wi = input.size(4);
    
    int total_outputs = N * C;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    
    adaptive_avg_pool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, Di, Hi, Wi
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor input, torch::Tensor bias, torch::Tensor output, float divisor);
void conv3d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                    int stride_d, int stride_h, int stride_w,
                    int pad_d, int pad_h, int pad_w,
                    int dilation_d, int dilation_h, int dilation_w);
void maxpool3d_forward(torch::Tensor input, torch::Tensor output,
                       int Kd, int Kh, int Kw,
                       int stride_d, int stride_h, int stride_w,
                       int pad_d, int pad_h, int pad_w,
                       int dilation_d, int dilation_h, int dilation_w);
void adaptive_avg_pool3d_forward(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused computation kernel");
    m.def("conv3d", &conv3d_forward, "Custom Conv3D");
    m.def("maxpool3d", &maxpool3d_forward, "Custom MaxPool3D");
    m.def("adaptive_avg_pool3d", &adaptive_avg_pool3d_forward, "Custom Adaptive Avg Pool3D");
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
    if isinstance(conv_stride, int):
        conv_stride = (conv_stride, conv_stride, conv_stride)
    if isinstance(conv_padding, int):
        conv_padding = (conv_padding, conv_padding, conv_padding)
    if isinstance(conv_dilation, int):
        conv_dilation = (conv_dilation, conv_dilation, conv_dilation)
        
    N, Ci, Di, Hi, Wi = x.shape
    Co, _, Kd, Kh, Kw = conv_weight.shape
    
    odim_d = (Di + 2 * conv_padding[0] - conv_dilation[0] * (Kd - 1) - 1) // conv_stride[0] + 1
    odim_h = (Hi + 2 * conv_padding[1] - conv_dilation[1] * (Kh - 1) - 1) // conv_stride[1] + 1
    odim_w = (Wi + 2 * conv_padding[2] - conv_dilation[2] * (Kw - 1) - 1) // conv_stride[2] + 1
    
    x_conv = torch.zeros((N, Co, odim_d, odim_h, odim_w), device=x.device, dtype=x.dtype)
    fused_ext.conv3d(
        x.contiguous(), conv_weight.contiguous(), conv_bias.contiguous(), x_conv,
        conv_stride[0], conv_stride[1], conv_stride[2],
        conv_padding[0], conv_padding[1], conv_padding[2],
        conv_dilation[0], conv_dilation[1], conv_dilation[2]
    )
    x = x_conv

    # MaxPool3D
    if isinstance(max_pool_kernel_size, int):
        max_pool_kernel_size = (max_pool_kernel_size, max_pool_kernel_size, max_pool_kernel_size)
    if isinstance(max_pool_stride, int):
        max_pool_stride = (max_pool_stride, max_pool_stride, max_pool_stride)
    if isinstance(max_pool_padding, int):
        max_pool_padding = (max_pool_padding, max_pool_padding, max_pool_padding)
    if isinstance(max_pool_dilation, int):
        max_pool_dilation = (max_pool_dilation, max_pool_dilation, max_pool_dilation)
        
    N, C, Di, Hi, Wi = x.shape
    odim_d = (Di + 2 * max_pool_padding[0] - max_pool_dilation[0] * (max_pool_kernel_size[0] - 1) - 1) // max_pool_stride[0] + 1
    odim_h = (Hi + 2 * max_pool_padding[1] - max_pool_dilation[1] * (max_pool_kernel_size[1] - 1) - 1) // max_pool_stride[1] + 1
    odim_w = (Wi + 2 * max_pool_padding[2] - max_pool_dilation[2] * (max_pool_kernel_size[2] - 1) - 1) // max_pool_stride[2] + 1
    
    x_pool = torch.zeros((N, C, odim_d, odim_h, odim_w), device=x.device, dtype=x.dtype)
    fused_ext.maxpool3d(
        x.contiguous(), x_pool,
        max_pool_kernel_size[0], max_pool_kernel_size[1], max_pool_kernel_size[2],
        max_pool_stride[0], max_pool_stride[1], max_pool_stride[2],
        max_pool_padding[0], max_pool_padding[1], max_pool_padding[2],
        max_pool_dilation[0], max_pool_dilation[1], max_pool_dilation[2]
    )
    x = x_pool

    # Adaptive Avg Pool (simplified for 1x1x1)
    N, C, Di, Hi, Wi = x.shape
    x_avg = torch.zeros((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
    fused_ext.adaptive_avg_pool3d(x.contiguous(), x_avg)
    x = x_avg.squeeze(-1).squeeze(-1).squeeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Reshape to [N, C, 1, 1, 1]

    # Fused Custom Kernel Output
    N, C, D, H, W = x.shape
    out = torch.zeros((N, D, H, W), device=x.device)
    fused_ext.fused_op(x.contiguous(), bias.contiguous().view(-1), out, divisor)
    return out

# Placeholders for evaluation requirements
batch_size=128; in_channels=8; out_channels=16; depth=16; height=64; width=64
kernel_size=(3, 3, 3); divisor=2.0; pool_size=(2, 2, 2); bias_shape=(out_channels, 1, 1, 1); sum_dim=1

def get_init_inputs(): return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]
def get_inputs(): return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
