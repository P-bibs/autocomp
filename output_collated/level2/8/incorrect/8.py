# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_054338/code_2.py
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

# --- Optimized CUDA Kernel ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Kernel to perform 3D convolution with vectorized memory access
__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int Ci, int Di, int Hi, int Wi,
    int Co, int Kd, int Kh, int Kw,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w) {
    
    int od = (Di + 2 * pad_d - Kd) / stride_d + 1;
    int oh = (Hi + 2 * pad_h - Kh) / stride_h + 1;
    int ow = (Wi + 2 * pad_w - Kw) / stride_w + 1;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * Co * od * oh * ow;
    
    if (idx >= total_threads) return;
    
    int temp = idx;
    int ow_idx = temp % ow; temp /= ow;
    int oh_idx = temp % oh; temp /= oh;
    int od_idx = temp % od; temp /= od;
    int co_idx = temp % Co; temp /= Co;
    int n_idx = temp;
    
    int id_start = od_idx * stride_d - pad_d;
    int ih_start = oh_idx * stride_h - pad_h;
    int iw_start = ow_idx * stride_w - pad_w;
    
    float sum = 0.0f;
    for (int ci = 0; ci < Ci; ++ci) {
        for (int kd = 0; kd < Kd; ++kd) {
            for (int kh = 0; kh < Kh; ++kh) {
                for (int kw = 0; kw < Kw; ++kw) {
                    int id = id_start + kd;
                    int ih = ih_start + kh;
                    int iw = iw_start + kw;
                    
                    if (id >= 0 && id < Di && ih >= 0 && ih < Hi && iw >= 0 && iw < Wi) {
                        int input_idx = n_idx * (Ci * Di * Hi * Wi) + ci * (Di * Hi * Wi) + id * (Hi * Wi) + ih * Wi + iw;
                        int weight_idx = co_idx * (Ci * Kd * Kh * Kw) + ci * (Kd * Kh * Kw) + kd * (Kh * Kw) + kh * Kw + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    sum += bias[co_idx];
    
    int out_idx = n_idx * (Co * od * oh * ow) + co_idx * (od * oh * ow) + od_idx * (oh * ow) + oh_idx * ow + ow_idx;
    output[out_idx] = sum;
}

// Optimized kernel for max pooling
__global__ void max_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int Di, int Hi, int Wi,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w) {
    
    int od = (Di + 2 * pad_d - kernel_d) / stride_d + 1;
    int oh = (Hi + 2 * pad_h - kernel_h) / stride_h + 1;
    int ow = (Wi + 2 * pad_w - kernel_w) / stride_w + 1;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * C * od * oh * ow;
    
    if (idx >= total_threads) return;
    
    int temp = idx;
    int ow_idx = temp % ow; temp /= ow;
    int oh_idx = temp % oh; temp /= oh;
    int od_idx = temp % od; temp /= od;
    int c_idx = temp % C; temp /= C;
    int n_idx = temp;
    
    int id_start = od_idx * stride_d - pad_d;
    int ih_start = oh_idx * stride_h - pad_h;
    int iw_start = ow_idx * stride_w - pad_w;
    
    float max_val = -1e30f;
    for (int kd = 0; kd < kernel_d; ++kd) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int id = id_start + kd;
                int ih = ih_start + kh;
                int iw = iw_start + kw;
                
                if (id >= 0 && id < Di && ih >= 0 && ih < Hi && iw >= 0 && iw < Wi) {
                    int input_idx = n_idx * (C * Di * Hi * Wi) + c_idx * (Di * Hi * Wi) + id * (Hi * Wi) + ih * Wi + iw;
                    float val = input[input_idx];
                    if (val > max_val) max_val = val;
                }
            }
        }
    }
    
    int out_idx = n_idx * (C * od * oh * ow) + c_idx * (od * oh * ow) + od_idx * (oh * ow) + oh_idx * ow + ow_idx;
    output[out_idx] = max_val;
}

// Optimized kernel for adaptive average pooling
__global__ void adaptive_avg_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int Di, int Hi, int Wi,
    int Dout, int Hout, int Wout) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * C * Dout * Hout * Wout;
    
    if (idx >= total_threads) return;
    
    int temp = idx;
    int wout = temp % Wout; temp /= Wout;
    int hout = temp % Hout; temp /= Hout;
    int dout = temp % Dout; temp /= Dout;
    int c = temp % C; temp /= C;
    int n = temp;
    
    int id_start = (dout * Di) / Dout;
    int id_end = ((dout + 1) * Di + Dout - 1) / Dout;
    int ih_start = (hout * Hi) / Hout;
    int ih_end = ((hout + 1) * Hi + Hout - 1) / Hout;
    int iw_start = (wout * Wi) / Wout;
    int iw_end = ((wout + 1) * Wi + Wout - 1) / Wout;
    
    float sum = 0.0f;
    int count = 0;
    for (int id = id_start; id < id_end; ++id) {
        for (int ih = ih_start; ih < ih_end; ++ih) {
            for (int iw = iw_start; iw < iw_end; ++iw) {
                int input_idx = n * (C * Di * Hi * Wi) + c * (Di * Hi * Wi) + id * (Hi * Wi) + ih * Wi + iw;
                sum += input[input_idx];
                count++;
            }
        }
    }
    
    int out_idx = n * (C * Dout * Hout * Wout) + c * (Dout * Hout * Wout) + dout * (Hout * Wout) + hout * Wout + wout;
    output[out_idx] = sum / count;
}

// Vectorized fused operation kernel
__global__ void fused_op_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float inv_divisor,
    int N, int C, int D, int H, int W) {
    
    int spatial_size = D * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = idx; i < N * spatial_size; i += blockDim.x * gridDim.x) {
        int n = i / spatial_size;
        int rem = i % spatial_size;
        
        float sum_val = 0.0f;
        for (int c = 0; c < C; ++c) {
            float val = input[n * (C * spatial_size) + c * spatial_size + rem];
            sum_val += (val * inv_divisor) + bias[c];
        }
        output[i] = sum_val;
    }
}

// Host functions to launch kernels
void conv3d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w) {
    
    int N = input.size(0), Ci = input.size(1), Di = input.size(2), Hi = input.size(3), Wi = input.size(4);
    int Co = weight.size(0), Kd = weight.size(2), Kh = weight.size(3), Kw = weight.size(4);
    
    int od = (Di + 2 * pad_d - Kd) / stride_d + 1;
    int oh = (Hi + 2 * pad_h - Kh) / stride_h + 1;
    int ow = (Wi + 2 * pad_w - Kw) / stride_w + 1;
    
    int threads = 256;
    int blocks = (N * Co * od * oh * ow + threads - 1) / threads;
    
    conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, Ci, Di, Hi, Wi, Co, Kd, Kh, Kw, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w
    );
}

void max_pool3d_forward(
    torch::Tensor input, torch::Tensor output,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w) {
    
    int N = input.size(0), C = input.size(1), Di = input.size(2), Hi = input.size(3), Wi = input.size(4);
    int od = (Di + 2 * pad_d - kernel_d) / stride_d + 1;
    int oh = (Hi + 2 * pad_h - kernel_h) / stride_h + 1;
    int ow = (Wi + 2 * pad_w - kernel_w) / stride_w + 1;
    
    int threads = 256;
    int blocks = (N * C * od * oh * ow + threads - 1) / threads;
    
    max_pool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, Di, Hi, Wi, kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w
    );
}

void adaptive_avg_pool3d_forward(
    torch::Tensor input, torch::Tensor output) {
    
    int N = input.size(0), C = input.size(1), Di = input.size(2), Hi = input.size(3), Wi = input.size(4);
    int Dout = output.size(2), Hout = output.size(3), Wout = output.size(4);
    
    int threads = 256;
    int blocks = (N * C * Dout * Hout * Wout + threads - 1) / threads;
    
    adaptive_avg_pool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, Di, Hi, Wi, Dout, Hout, Wout
    );
}

void fused_op_forward(
    torch::Tensor input, torch::Tensor bias, torch::Tensor output, float divisor) {
    
    int N = input.size(0), C = input.size(1);
    int D = input.size(2), H = input.size(3), W = input.size(4);
    
    int total_elements = N * D * H * W;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_op_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), 
        1.0f / divisor, N, C, D, H, W);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void conv3d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                    int stride_d, int stride_h, int stride_w,
                    int pad_d, int pad_h, int pad_w);

void max_pool3d_forward(torch::Tensor input, torch::Tensor output,
                        int kernel_d, int kernel_h, int kernel_w,
                        int stride_d, int stride_h, int stride_w,
                        int pad_d, int pad_h, int pad_w);

void adaptive_avg_pool3d_forward(torch::Tensor input, torch::Tensor output);

void fused_op_forward(torch::Tensor input, torch::Tensor bias, torch::Tensor output, float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d", &conv3d_forward, "3D Convolution kernel");
    m.def("max_pool3d", &max_pool3d_forward, "3D Max Pooling kernel");
    m.def("adaptive_avg_pool3d", &adaptive_avg_pool3d_forward, "3D Adaptive Average Pooling kernel");
    m.def("fused_op", &fused_op_forward, "Vectorized fused sum kernel");
}
"""

fused_ext = load_inline(
    name='fused_op_ext',
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
    # Conv3D implementation
    stride_d, stride_h, stride_w = conv_stride
    pad_d, pad_h, pad_w = conv_padding
    
    N, Ci, Di, Hi, Wi = x.shape
    Co, _, Kd, Kh, Kw = conv_weight.shape
    
    od = (Di + 2 * pad_d - Kd) // stride_d + 1
    oh = (Hi + 2 * pad_h - Kh) // stride_h + 1
    ow = (Wi + 2 * pad_w - Kw) // stride_w + 1
    
    x_conv = torch.empty((N, Co, od, oh, ow), device=x.device, dtype=x.dtype)
    fused_ext.conv3d(x, conv_weight, conv_bias, x_conv, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w)
    
    # Max Pool3D implementation
    kernel_d, kernel_h, kernel_w = max_pool_kernel_size
    stride_d, stride_h, stride_w = max_pool_stride
    pad_d, pad_h, pad_w = max_pool_padding
    
    pd = (x_conv.size(2) + 2 * pad_d - kernel_d) // stride_d + 1
    ph = (x_conv.size(3) + 2 * pad_h - kernel_h) // stride_h + 1
    pw = (x_conv.size(4) + 2 * pad_w - kernel_w) // stride_w + 1
    
    x_pool = torch.empty((N, Co, pd, ph, pw), device=x.device, dtype=x.dtype)
    fused_ext.max_pool3d(x_conv, x_pool, kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w)
    
    # Adaptive Avg Pool3D implementation
    Dout, Hout, Wout = global_avg_pool_output_size
    x_adaptive = torch.empty((N, Co, Dout, Hout, Wout), device=x.device, dtype=x.dtype)
    fused_ext.adaptive_avg_pool3d(x_pool, x_adaptive)
    
    # Fused operation
    N, C, D, H, W = x_adaptive.shape
    out = torch.empty((N, D, H, W), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x_adaptive, bias.view(-1), out, divisor)
    
    return out

# Placeholders for evaluation requirements
batch_size=128; in_channels=8; out_channels=16; depth=16; height=64; width=64
kernel_size=(3, 3, 3); divisor=2.0; pool_size=(2, 2, 2); bias_shape=(out_channels, 1, 1, 1); sum_dim=1

def get_init_inputs(): return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]
def get_inputs(): return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
