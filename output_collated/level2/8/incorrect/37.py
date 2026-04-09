# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_061658/code_3.py
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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * Co * ((Di + 2 * pad_d - dilation_d * (Kd - 1) - 1) / stride_d + 1) *
                                   ((Hi + 2 * pad_h - dilation_h * (Kh - 1) - 1) / stride_h + 1) *
                                   ((Wi + 2 * pad_w - dilation_w * (Kw - 1) - 1) / stride_w + 1);

    if (tid < total_threads) {
        int o_w = ((Wi + 2 * pad_w - dilation_w * (Kw - 1) - 1) / stride_w + 1);
        int o_h = ((Hi + 2 * pad_h - dilation_h * (Kh - 1) - 1) / stride_h + 1);
        int o_d = ((Di + 2 * pad_d - dilation_d * (Kd - 1) - 1) / stride_d + 1);

        int n = tid / (Co * o_d * o_h * o_w);
        int co = (tid / (o_d * o_h * o_w)) % Co;
        int od = (tid / (o_h * o_w)) % o_d;
        int oh = (tid / o_w) % o_h;
        int ow = tid % o_w;

        int id = od * stride_d - pad_d;
        int ih = oh * stride_h - pad_h;
        int iw = ow * stride_w - pad_w;

        float sum = 0.0f;
        #pragma unroll
        for (int ci = 0; ci < Ci; ++ci) {
            #pragma unroll
            for (int kd = 0; kd < Kd; ++kd) {
                #pragma unroll
                for (int kh = 0; kh < Kh; ++kh) {
                    #pragma unroll
                    for (int kw = 0; kw < Kw; ++kw) {
                        int in_d = id + kd * dilation_d;
                        int in_h = ih + kh * dilation_h;
                        int in_w = iw + kw * dilation_w;

                        if (in_d >= 0 && in_d < Di && in_h >= 0 && in_h < Hi && in_w >= 0 && in_w < Wi) {
                            int input_idx = n * (Ci * Di * Hi * Wi) + ci * (Di * Hi * Wi) + in_d * (Hi * Wi) + in_h * Wi + in_w;
                            int weight_idx = co * (Ci * Kd * Kh * Kw) + ci * (Kd * Kh * Kw) + kd * (Kh * Kw) + kh * Kw + kw;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        int output_idx = n * (Co * o_d * o_h * o_w) + co * (o_d * o_h * o_w) + od * (o_h * o_w) + oh * o_w + ow;
        output[output_idx] = sum + bias[co];
    }
}

__global__ void maxpool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int Di, int Hi, int Wi,
    int Kd, int Kh, int Kw,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int o_d = ((Di + 2 * pad_d - dilation_d * (Kd - 1) - 1) / stride_d + 1);
    int o_h = ((Hi + 2 * pad_h - dilation_h * (Kh - 1) - 1) / stride_h + 1);
    int o_w = ((Wi + 2 * pad_w - dilation_w * (Kw - 1) - 1) / stride_w + 1);
    int total_threads = N * C * o_d * o_h * o_w;

    if (tid < total_threads) {
        int n = tid / (C * o_d * o_h * o_w);
        int c = (tid / (o_d * o_h * o_w)) % C;
        int od = (tid / (o_h * o_w)) % o_d;
        int oh = (tid / o_w) % o_h;
        int ow = tid % o_w;

        int id_start = od * stride_d - pad_d;
        int ih_start = oh * stride_h - pad_h;
        int iw_start = ow * stride_w - pad_w;

        float max_val = -FLT_MAX;
        #pragma unroll
        for (int kd = 0; kd < Kd; ++kd) {
            #pragma unroll
            for (int kh = 0; kh < Kh; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < Kw; ++kw) {
                    int in_d = id_start + kd * dilation_d;
                    int in_h = ih_start + kh * dilation_h;
                    int in_w = iw_start + kw * dilation_w;

                    if (in_d >= 0 && in_d < Di && in_h >= 0 && in_h < Hi && in_w >= 0 && in_w < Wi) {
                        int input_idx = n * (C * Di * Hi * Wi) + c * (Di * Hi * Wi) + in_d * (Hi * Wi) + in_h * Wi + in_w;
                        if (input[input_idx] > max_val) {
                            max_val = input[input_idx];
                        }
                    }
                }
            }
        }
        int output_idx = n * (C * o_d * o_h * o_w) + c * (o_d * o_h * o_w) + od * (o_h * o_w) + oh * o_w + ow;
        output[output_idx] = max_val;
    }
}

__global__ void adaptive_avg_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int Di, int Hi, int Wi,
    int Do, int Ho, int Wo
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * C * Do * Ho * Wo;

    if (tid < total_threads) {
        int n = tid / (C * Do * Ho * Wo);
        int c = (tid / (Do * Ho * Wo)) % C;
        int od = (tid / (Ho * Wo)) % Do;
        int oh = (tid / Wo) % Ho;
        int ow = tid % Wo;

        int id_start = (od * Di) / Do;
        int id_end = ((od + 1) * Di + Do - 1) / Do;
        int ih_start = (oh * Hi) / Ho;
        int ih_end = ((oh + 1) * Hi + Ho - 1) / Ho;
        int iw_start = (ow * Wi) / Wo;
        int iw_end = ((ow + 1) * Wi + Wo - 1) / Wo;

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
        int output_idx = n * (C * Do * Ho * Wo) + c * (Do * Ho * Wo) + od * (Ho * Wo) + oh * Wo + ow;
        output[output_idx] = sum / count;
    }
}

__global__ void fused_op_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float divisor,
    int N, int C, int D, int H, int W) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = D * H * W;
    int total_elements = N * spatial_size;

    if (tid < total_elements) {
        int n = tid / spatial_size;
        int rem = tid % spatial_size;

        float sum_val = 0.0f;
        // Optimization: Loop unrolling to reduce branch overhead 
        // and improve instruction throughput.
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            int input_idx = (n * C + c) * spatial_size + rem;
            sum_val += (input[input_idx] / divisor) + bias[c];
        }
        output[tid] = sum_val;
    }
}

void conv3d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, 
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w) {
    
    int N = input.size(0);
    int Ci = input.size(1);
    int Di = input.size(2);
    int Hi = input.size(3);
    int Wi = input.size(4);
    
    int Co = weight.size(0);
    int Kd = weight.size(2);
    int Kh = weight.size(3);
    int Kw = weight.size(4);
    
    int o_d = ((Di + 2 * pad_d - dilation_d * (Kd - 1) - 1) / stride_d + 1);
    int o_h = ((Hi + 2 * pad_h - dilation_h * (Kh - 1) - 1) / stride_h + 1);
    int o_w = ((Wi + 2 * pad_w - dilation_w * (Kw - 1) - 1) / stride_w + 1);
    
    int total_threads = N * Co * o_d * o_h * o_w;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;
    
    conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, Ci, Di, Hi, Wi, Co, Kd, Kh, Kw,
        stride_d, stride_h, stride_w, pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w);
}

void maxpool3d_forward(
    torch::Tensor input, torch::Tensor output,
    int Kd, int Kh, int Kw,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w) {
    
    int N = input.size(0);
    int C = input.size(1);
    int Di = input.size(2);
    int Hi = input.size(3);
    int Wi = input.size(4);
    
    int o_d = ((Di + 2 * pad_d - dilation_d * (Kd - 1) - 1) / stride_d + 1);
    int o_h = ((Hi + 2 * pad_h - dilation_h * (Kh - 1) - 1) / stride_h + 1);
    int o_w = ((Wi + 2 * pad_w - dilation_w * (Kw - 1) - 1) / stride_w + 1);
    
    int total_threads = N * C * o_d * o_h * o_w;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;
    
    maxpool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, Di, Hi, Wi, Kd, Kh, Kw,
        stride_d, stride_h, stride_w, pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w);
}

void adaptive_avg_pool3d_forward(
    torch::Tensor input, torch::Tensor output,
    int Do, int Ho, int Wo) {
    
    int N = input.size(0);
    int C = input.size(1);
    int Di = input.size(2);
    int Hi = input.size(3);
    int Wi = input.size(4);
    
    int total_threads = N * C * Do * Ho * Wo;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;
    
    adaptive_avg_pool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, Di, Hi, Wi, Do, Ho, Wo);
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
"""

cpp_source = r"""
#include <torch/extension.h>

void conv3d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, 
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w);

void maxpool3d_forward(
    torch::Tensor input, torch::Tensor output,
    int Kd, int Kh, int Kw,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w);

void adaptive_avg_pool3d_forward(
    torch::Tensor input, torch::Tensor output,
    int Do, int Ho, int Wo);

void fused_op_forward(torch::Tensor input, torch::Tensor bias, torch::Tensor output, float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d", &conv3d_forward, "Custom Conv3D kernel");
    m.def("maxpool3d", &maxpool3d_forward, "Custom MaxPool3D kernel");
    m.def("adaptive_avg_pool3d", &adaptive_avg_pool3d_forward, "Custom AdaptiveAvgPool3D kernel");
    m.def("fused_op", &fused_op_forward, "Fused divide, bias, and sum kernel with loop unrolling");
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
    N, Ci, Di, Hi, Wi = x.shape
    Co, _, Kd, Kh, Kw = conv_weight.shape
    stride_d, stride_h, stride_w = conv_stride
    pad_d, pad_h, pad_w = conv_padding
    dilation_d, dilation_h, dilation_w = conv_dilation
    
    o_d = ((Di + 2 * pad_d - dilation_d * (Kd - 1) - 1) // stride_d + 1)
    o_h = ((Hi + 2 * pad_h - dilation_h * (Kh - 1) - 1) // stride_h + 1)
    o_w = ((Wi + 2 * pad_w - dilation_w * (Kw - 1) - 1) // stride_w + 1)
    
    x_conv = torch.zeros((N, Co, o_d, o_h, o_w), device=x.device, dtype=x.dtype)
    fused_ext.conv3d(x.contiguous(), conv_weight.contiguous(), conv_bias.contiguous(), 
                     x_conv, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, 
                     dilation_d, dilation_h, dilation_w)
    
    # MaxPool3D
    Kd_mp, Kh_mp, Kw_mp = max_pool_kernel_size
    stride_d_mp, stride_h_mp, stride_w_mp = max_pool_stride
    pad_d_mp, pad_h_mp, pad_w_mp = max_pool_padding
    dilation_d_mp, dilation_h_mp, dilation_w_mp = max_pool_dilation
    
    o_d_mp = ((x_conv.size(2) + 2 * pad_d_mp - dilation_d_mp * (Kd_mp - 1) - 1) // stride_d_mp + 1)
    o_h_mp = ((x_conv.size(3) + 2 * pad_h_mp - dilation_h_mp * (Kh_mp - 1) - 1) // stride_h_mp + 1)
    o_w_mp = ((x_conv.size(4) + 2 * pad_w_mp - dilation_w_mp * (Kw_mp - 1) - 1) // stride_w_mp + 1)
    
    x_pool = torch.zeros((N, Co, o_d_mp, o_h_mp, o_w_mp), device=x.device, dtype=x.dtype)
    fused_ext.maxpool3d(x_conv.contiguous(), x_pool, 
                        Kd_mp, Kh_mp, Kw_mp, 
                        stride_d_mp, stride_h_mp, stride_w_mp, 
                        pad_d_mp, pad_h_mp, pad_w_mp, 
                        dilation_d_mp, dilation_h_mp, dilation_w_mp)
    
    # AdaptiveAvgPool3D
    Do, Ho, Wo = global_avg_pool_output_size
    x_adapt = torch.zeros((N, Co, Do, Ho, Wo), device=x.device, dtype=x.dtype)
    fused_ext.adaptive_avg_pool3d(x_pool.contiguous(), x_adapt, Do, Ho, Wo)
    
    # Fused Custom Kernel Output
    out = torch.zeros((N, Do, Ho, Wo), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x_adapt.contiguous(), bias.contiguous().view(-1), out, divisor)
    return out

# Placeholders for evaluation requirements
batch_size=128; in_channels=8; out_channels=16; depth=16; height=64; width=64
kernel_size=(3, 3, 3); divisor=2.0; pool_size=(2, 2, 2); bias_shape=(out_channels, 1, 1, 1); sum_dim=1

def get_init_inputs(): return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]
def get_inputs(): return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
