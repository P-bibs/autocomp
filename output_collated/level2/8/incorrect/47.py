# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_064410/code_0.py
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

__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * D_out * H_out * W_out;
    
    if (idx < total_elements) {
        int w_out_idx = idx % W_out;
        int h_out_idx = (idx / W_out) % H_out;
        int d_out_idx = (idx / (W_out * H_out)) % D_out;
        int c_out_idx = (idx / (W_out * H_out * D_out)) % C_out;
        int n_idx = idx / (W_out * H_out * D_out * C_out);
        
        int d_in_start = d_out_idx * stride_d - padding_d;
        int h_in_start = h_out_idx * stride_h - padding_h;
        int w_in_start = w_out_idx * stride_w - padding_w;
        
        float sum = 0.0f;
        
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kd = 0; kd < kernel_d; ++kd) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int d_in = d_in_start + kd;
                        int h_in = h_in_start + kh;
                        int w_in = w_in_start + kw;
                        
                        if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            int input_idx = n_idx * (C_in * D_in * H_in * W_in) + 
                                          c_in * (D_in * H_in * W_in) + 
                                          d_in * (H_in * W_in) + 
                                          h_in * W_in + w_in;
                            
                            int weight_idx = c_out_idx * (C_in * kernel_d * kernel_h * kernel_w) + 
                                           c_in * (kernel_d * kernel_h * kernel_w) + 
                                           kd * (kernel_h * kernel_w) + 
                                           kh * kernel_w + kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        int output_idx = idx;
        output[output_idx] = sum + bias[c_out_idx];
    }
}

__global__ void max_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * D_out * H_out * W_out;
    
    if (idx < total_elements) {
        int w_out_idx = idx % W_out;
        int h_out_idx = (idx / W_out) % H_out;
        int d_out_idx = (idx / (W_out * H_out)) % D_out;
        int c_idx = (idx / (W_out * H_out * D_out)) % C;
        int n_idx = idx / (W_out * H_out * D_out * C);
        
        int d_start = d_out_idx * stride_d - padding_d;
        int h_start = h_out_idx * stride_h - padding_h;
        int w_start = w_out_idx * stride_w - padding_w;
        
        float max_val = -FLT_MAX;
        
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int d_in = d_start + kd;
                    int h_in = h_start + kh;
                    int w_in = w_start + kw;
                    
                    if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int input_idx = n_idx * (C * D_in * H_in * W_in) + 
                                      c_idx * (D_in * H_in * W_in) + 
                                      d_in * (H_in * W_in) + 
                                      h_in * W_in + w_in;
                        
                        max_val = fmaxf(max_val, input[input_idx]);
                    }
                }
            }
        }
        
        output[idx] = max_val;
    }
}

__global__ void adaptive_avg_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * D_out * H_out * W_out;
    
    if (idx < total_elements) {
        int w_out_idx = idx % W_out;
        int h_out_idx = (idx / W_out) % H_out;
        int d_out_idx = (idx / (W_out * H_out)) % D_out;
        int c_idx = (idx / (W_out * H_out * D_out)) % C;
        int n_idx = idx / (W_out * H_out * D_out * C);
        
        int d_start = (d_out_idx * D_in) / D_out;
        int d_end = ((d_out_idx + 1) * D_in + D_out - 1) / D_out;
        int h_start = (h_out_idx * H_in) / H_out;
        int h_end = ((h_out_idx + 1) * H_in + H_out - 1) / H_out;
        int w_start = (w_out_idx * W_in) / W_out;
        int w_end = ((w_out_idx + 1) * W_in + W_out - 1) / W_out;
        
        float sum = 0.0f;
        int count = 0;
        
        for (int d = d_start; d < d_end; ++d) {
            for (int h = h_start; h < h_end; ++h) {
                for (int w = w_start; w < w_end; ++w) {
                    int input_idx = n_idx * (C * D_in * H_in * W_in) + 
                                  c_idx * (D_in * H_in * W_in) + 
                                  d * (H_in * W_in) + 
                                  h * W_in + w;
                    sum += input[input_idx];
                    count++;
                }
            }
        }
        
        output[idx] = sum / count;
    }
}

__global__ void fused_op_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias_fused,
    float* __restrict__ output,
    float divisor,
    int N, int C, int D, int H, int W) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * D * H * W;

    if (idx < total_elements) {
        int n = idx / (D * H * W);
        int d = (idx / (H * W)) % D;
        int h = (idx / W) % H;
        int w = idx % W;

        float sum_val = 0.0f;
        for (int c = 0; c < C; ++c) {
            int input_idx = n * (C * D * H * W) + c * (D * H * W) + d * (H * W) + h * W + w;
            sum_val += (input[input_idx] / divisor) + bias_fused[c];
        }
        output[idx] = sum_val;
    }
}

void conv3d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, 
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w) {
    
    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    
    int C_out = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    
    int D_out = output.size(2);
    int H_out = output.size(3);
    int W_out = output.size(4);
    
    int threads = 256;
    int blocks = (N * C_out * D_out * H_out * W_out + threads - 1) / threads;
    
    conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in, C_out, D_out, H_out, W_out,
        kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w);
}

void max_pool3d_forward(
    torch::Tensor input, torch::Tensor output,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w) {
    
    int N = input.size(0);
    int C = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    
    int D_out = output.size(2);
    int H_out = output.size(3);
    int W_out = output.size(4);
    
    int threads = 256;
    int blocks = (N * C * D_out * H_out * W_out + threads - 1) / threads;
    
    max_pool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, D_in, H_in, W_in, D_out, H_out, W_out,
        kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w);
}

void adaptive_avg_pool3d_forward(
    torch::Tensor input, torch::Tensor output) {
    
    int N = input.size(0);
    int C = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    
    int D_out = output.size(2);
    int H_out = output.size(3);
    int W_out = output.size(4);
    
    int threads = 256;
    int blocks = (N * C * D_out * H_out * W_out + threads - 1) / threads;
    
    adaptive_avg_pool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, D_in, H_in, W_in, D_out, H_out, W_out);
}

void fused_op_forward(torch::Tensor input, torch::Tensor bias, torch::Tensor output, float divisor) {
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    
    int threads = 256;
    int blocks = (N * D * H * W + threads - 1) / threads;
    
    fused_op_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), 
        divisor, N, C, D, H, W);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void conv3d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                   torch::Tensor output, 
                   int stride_d, int stride_h, int stride_w,
                   int padding_d, int padding_h, int padding_w);

void max_pool3d_forward(torch::Tensor input, torch::Tensor output,
                       int kernel_d, int kernel_h, int kernel_w,
                       int stride_d, int stride_h, int stride_w,
                       int padding_d, int padding_h, int padding_w);

void adaptive_avg_pool3d_forward(torch::Tensor input, torch::Tensor output);

void fused_op_forward(torch::Tensor input, torch::Tensor bias, torch::Tensor output, float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d", &conv3d_forward, "Conv3d kernel");
    m.def("max_pool3d", &max_pool3d_forward, "MaxPool3d kernel");
    m.def("adaptive_avg_pool3d", &adaptive_avg_pool3d_forward, "AdaptiveAvgPool3d kernel");
    m.def("fused_op", &fused_op_forward, "Fused divide, bias, and sum kernel");
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
    # Replace F.conv3d with custom CUDA implementation
    N, C_in, D_in, H_in, W_in = x.shape
    C_out, _, kernel_d, kernel_h, kernel_w = conv_weight.shape
    
    # Calculate output dimensions for conv3d
    stride_d, stride_h, stride_w = conv_stride
    padding_d, padding_h, padding_w = conv_padding
    
    D_out = (D_in + 2*padding_d - kernel_d) // stride_d + 1
    H_out = (H_in + 2*padding_h - kernel_h) // stride_h + 1
    W_out = (W_in + 2*padding_w - kernel_w) // stride_w + 1
    
    conv_output = torch.zeros((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Launch custom conv3d kernel
    fused_ext.conv3d(x.contiguous(), conv_weight.contiguous(), conv_bias.contiguous(),
                     conv_output, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w)
    
    # Replace F.max_pool3d with custom CUDA implementation
    pool_d, pool_h, pool_w = max_pool_kernel_size
    pool_stride_d, pool_stride_h, pool_stride_w = max_pool_stride if max_pool_stride else (pool_d, pool_h, pool_w)
    pool_pad_d, pool_pad_h, pool_pad_w = max_pool_padding
    
    # Calculate output dimensions for max pooling
    D_pool_out = (D_out + 2*pool_pad_d - pool_d) // pool_stride_d + 1
    H_pool_out = (H_out + 2*pool_pad_h - pool_h) // pool_stride_h + 1
    W_pool_out = (W_out + 2*pool_pad_w - pool_w) // pool_stride_w + 1
    
    pool_output = torch.zeros((N, C_out, D_pool_out, H_pool_out, W_pool_out), device=x.device, dtype=x.dtype)
    
    # Launch custom max_pool3d kernel
    fused_ext.max_pool3d(conv_output.contiguous(), pool_output,
                         pool_d, pool_h, pool_w,
                         pool_stride_d, pool_stride_h, pool_stride_w,
                         pool_pad_d, pool_pad_h, pool_pad_w)
    
    # Replace F.adaptive_avg_pool3d with custom CUDA implementation
    global_d, global_h, global_w = global_avg_pool_output_size
    adaptive_output = torch.zeros((N, C_out, global_d, global_h, global_w), device=x.device, dtype=x.dtype)
    
    # Launch custom adaptive_avg_pool3d kernel
    fused_ext.adaptive_avg_pool3d(pool_output.contiguous(), adaptive_output)
    
    # Fused Custom Kernel Output
    # Shape of x after adaptive pool: [N, C, D, H, W]
    N_f, C_f, D_f, H_f, W_f = adaptive_output.shape
    out = torch.zeros((N_f, D_f, H_f, W_f), device=x.device)
    
    fused_ext.fused_op(adaptive_output.contiguous(), bias.contiguous().view(-1), out, divisor)
    return out

# Placeholders for evaluation requirements
batch_size=128; in_channels=8; out_channels=16; depth=16; height=64; width=64
kernel_size=(3, 3, 3); divisor=2.0; pool_size=(2, 2, 2); bias_shape=(out_channels, 1, 1, 1); sum_dim=1

def get_init_inputs(): return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]
def get_inputs(): return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
