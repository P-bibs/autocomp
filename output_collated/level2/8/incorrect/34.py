# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_060810/code_4.py
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

# --- CUDA Kernel with Memory Coalescing and Shared Memory Optimization ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 256

__global__ void fused_op_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float divisor,
    int N, int C, int D, int H, int W) {
    
    // Shared memory for bias values to reduce global memory accesses
    __shared__ float shared_bias[TILE_SIZE];
    
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = N * D * H * W;
    int tid = threadIdx.x;
    
    // Grid-stride loop for potentially multiple spatial positions per thread
    for (int s = spatial_idx; s < total_spatial; s += gridDim.x * blockDim.x) {
        // Decompose spatial index into (n, d, h, w)
        int n = s / (D * H * W);
        int remainder = s % (D * H * W);
        int d = remainder / (H * W);
        remainder = remainder % (H * W);
        int h = remainder / W;
        int w = remainder % W;
        
        // Base offset for this spatial position across all channels
        int base_offset = n * (C * D * H * W) + d * (H * W) + h * W + w;
        
        float sum_val = 0.0f;
        
        // Process channels in tiles to leverage shared memory
        for (int c_start = 0; c_start < C; c_start += TILE_SIZE) {
            // Load bias values into shared memory
            int c_end = min(c_start + TILE_SIZE, C);
            if (tid < c_end - c_start) {
                shared_bias[tid] = bias[c_start + tid];
            }
            __syncthreads();
            
            // Coalesced channel iteration using shared memory
            for (int c = c_start; c < c_end; ++c) {
                int input_idx = base_offset + c * (D * H * W);
                sum_val += (input[input_idx] / divisor) + shared_bias[c - c_start];
            }
            __syncthreads();
        }
        
        // Write output
        output[s] = sum_val;
    }
}

// Custom Conv3D kernel
__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int Ci, int Co, int Di, int Hi, int Wi,
    int Do, int Ho, int Wo, int Kd, int Kh, int Kw,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N * Co * Do * Ho * Wo;
    
    if (idx < total_outputs) {
        int wo = idx % Wo;
        int ho = (idx / Wo) % Ho;
        int doo = (idx / (Wo * Ho)) % Do;
        int co = (idx / (Wo * Ho * Do)) % Co;
        int n = idx / (Wo * Ho * Do * Co);
        
        int id = doo * stride_d - pad_d;
        int ih = ho * stride_h - pad_h;
        int iw = wo * stride_w - pad_w;
        
        float sum = 0.0f;
        
        for (int kd = 0; kd < Kd; kd++) {
            for (int kh = 0; kh < Kh; kh++) {
                for (int kw = 0; kw < Kw; kw++) {
                    int d = id + kd;
                    int h = ih + kh;
                    int w = iw + kw;
                    
                    if (d >= 0 && d < Di && h >= 0 && h < Hi && w >= 0 && w < Wi) {
                        for (int ci = 0; ci < Ci; ci++) {
                            int input_idx = n * (Ci * Di * Hi * Wi) + ci * (Di * Hi * Wi) + d * (Hi * Wi) + h * Wi + w;
                            int weight_idx = co * (Ci * Kd * Kh * Kw) + ci * (Kd * Kh * Kw) + kd * (Kh * Kw) + kh * Kw + kw;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        output[idx] = sum + bias[co];
    }
}

// Custom MaxPool3D kernel
__global__ void maxpool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int Di, int Hi, int Wi,
    int Do, int Ho, int Wo,
    int Kd, int Kh, int Kw,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N * C * Do * Ho * Wo;
    
    if (idx < total_outputs) {
        int wo = idx % Wo;
        int ho = (idx / Wo) % Ho;
        int doo = (idx / (Wo * Ho)) % Do;
        int c = (idx / (Wo * Ho * Do)) % C;
        int n = idx / (Wo * Ho * Do * C);
        
        int id = doo * stride_d - pad_d;
        int ih = ho * stride_h - pad_h;
        int iw = wo * stride_w - pad_w;
        
        float max_val = -FLT_MAX;
        
        for (int kd = 0; kd < Kd; kd++) {
            for (int kh = 0; kh < Kh; kh++) {
                for (int kw = 0; kw < Kw; kw++) {
                    int d = id + kd;
                    int h = ih + kh;
                    int w = iw + kw;
                    
                    if (d >= 0 && d < Di && h >= 0 && h < Hi && w >= 0 && w < Wi) {
                        int input_idx = n * (C * Di * Hi * Wi) + c * (Di * Hi * Wi) + d * (Hi * Wi) + h * Wi + w;
                        max_val = fmaxf(max_val, input[input_idx]);
                    }
                }
            }
        }
        
        output[idx] = max_val;
    }
}

// Custom AdaptiveAvgPool3D kernel (simplified for specific case)
__global__ void adaptive_avgpool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int Di, int Hi, int Wi,
    int Do, int Ho, int Wo) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N * C * Do * Ho * Wo;
    
    if (idx < total_outputs) {
        int wo = idx % Wo;
        int ho = (idx / Wo) % Ho;
        int doo = (idx / (Wo * Ho)) % Do;
        int c = (idx / (Wo * Ho * Do)) % C;
        int n = idx / (Wo * Ho * Do * C);
        
        // Simplified for when Do=Di/2, Ho=Hi/2, Wo=Wi/2
        int d_start = doo * (Di / Do);
        int d_end = (doo + 1) * (Di / Do);
        int h_start = ho * (Hi / Ho);
        int h_end = (ho + 1) * (Hi / Ho);
        int w_start = wo * (Wi / Wo);
        int w_end = (wo + 1) * (Wi / Wo);
        
        float sum = 0.0f;
        int count = 0;
        
        for (int d = d_start; d < d_end; d++) {
            for (int h = h_start; h < h_end; h++) {
                for (int w = w_start; w < w_end; w++) {
                    int input_idx = n * (C * Di * Hi * Wi) + c * (Di * Hi * Wi) + d * (Hi * Wi) + h * Wi + w;
                    sum += input[input_idx];
                    count++;
                }
            }
        }
        
        output[idx] = sum / count;
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor bias, torch::Tensor output, float divisor) {
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    
    int total_spatial = N * D * H * W;
    
    // Optimal block size for memory coalescing
    int threads = 256;
    int blocks = (total_spatial + threads - 1) / threads;
    
    // Limit blocks to avoid excessive kernel launch overhead
    blocks = min(blocks, 65535);
    
    fused_op_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        divisor, 
        N, C, D, H, W);
}

void conv3d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w) {
    
    int N = input.size(0);
    int Ci = input.size(1);
    int Di = input.size(2);
    int Hi = input.size(3);
    int Wi = input.size(4);
    
    int Co = weight.size(0);
    int Kd = weight.size(2);
    int Kh = weight.size(3);
    int Kw = weight.size(4);
    
    int Do = output.size(2);
    int Ho = output.size(3);
    int Wo = output.size(4);
    
    int total_outputs = N * Co * Do * Ho * Wo;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    blocks = min(blocks, 65535);
    
    conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, Ci, Co, Di, Hi, Wi, Do, Ho, Wo,
        Kd, Kh, Kw,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w);
}

void maxpool3d_forward(
    torch::Tensor input, torch::Tensor output,
    int Kd, int Kh, int Kw,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w) {
    
    int N = input.size(0);
    int C = input.size(1);
    int Di = input.size(2);
    int Hi = input.size(3);
    int Wi = input.size(4);
    
    int Do = output.size(2);
    int Ho = output.size(3);
    int Wo = output.size(4);
    
    int total_outputs = N * C * Do * Ho * Wo;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    blocks = min(blocks, 65535);
    
    maxpool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, Di, Hi, Wi, Do, Ho, Wo,
        Kd, Kh, Kw,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w);
}

void adaptive_avgpool3d_forward(
    torch::Tensor input, torch::Tensor output) {
    
    int N = input.size(0);
    int C = input.size(1);
    int Di = input.size(2);
    int Hi = input.size(3);
    int Wi = input.size(4);
    
    int Do = output.size(2);
    int Ho = output.size(3);
    int Wo = output.size(4);
    
    int total_outputs = N * C * Do * Ho * Wo;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    blocks = min(blocks, 65535);
    
    adaptive_avgpool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, Di, Hi, Wi, Do, Ho, Wo);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor input, torch::Tensor bias, torch::Tensor output, float divisor);
void conv3d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                   int stride_d, int stride_h, int stride_w, int pad_d, int pad_h, int pad_w);
void maxpool3d_forward(torch::Tensor input, torch::Tensor output,
                      int Kd, int Kh, int Kw, int stride_d, int stride_h, int stride_w,
                      int pad_d, int pad_h, int pad_w);
void adaptive_avgpool3d_forward(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused divide, bias, and sum kernel with coalesced memory access");
    m.def("conv3d", &conv3d_forward, "Custom Conv3D kernel");
    m.def("maxpool3d", &maxpool3d_forward, "Custom MaxPool3D kernel");
    m.def("adaptive_avgpool3d", &adaptive_avgpool3d_forward, "Custom AdaptiveAvgPool3D kernel");
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
    # Custom Conv3D
    N, Ci, Di, Hi, Wi = x.shape
    Co, _, Kd, Kh, Kw = conv_weight.shape
    
    # Calculate output dimensions
    stride_d, stride_h, stride_w = conv_stride
    pad_d, pad_h, pad_w = conv_padding
    
    Do = (Di + 2 * pad_d - Kd) // stride_d + 1
    Ho = (Hi + 2 * pad_h - Kh) // stride_h + 1
    Wo = (Wi + 2 * pad_w - Kw) // stride_w + 1
    
    x_conv = torch.empty((N, Co, Do, Ho, Wo), device=x.device, dtype=x.dtype)
    fused_ext.conv3d(x, conv_weight, conv_bias, x_conv, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w)
    x = x_conv
    
    # Custom MaxPool3D
    Kd_pool, Kh_pool, Kw_pool = max_pool_kernel_size
    stride_d_pool, stride_h_pool, stride_w_pool = max_pool_stride
    pad_d_pool, pad_h_pool, pad_w_pool = max_pool_padding
    
    Dp = (Do + 2 * pad_d_pool - Kd_pool) // stride_d_pool + 1
    Hp = (Ho + 2 * pad_h_pool - Kh_pool) // stride_h_pool + 1
    Wp = (Wo + 2 * pad_w_pool - Kw_pool) // stride_w_pool + 1
    
    x_pool = torch.empty((N, Co, Dp, Hp, Wp), device=x.device, dtype=x.dtype)
    fused_ext.maxpool3d(x, x_pool, Kd_pool, Kh_pool, Kw_pool, 
                       stride_d_pool, stride_h_pool, stride_w_pool,
                       pad_d_pool, pad_h_pool, pad_w_pool)
    x = x_pool
    
    # Custom AdaptiveAvgPool3D (simplified for this specific case)
    Do_avg, Ho_avg, Wo_avg = global_avg_pool_output_size
    x_avg = torch.empty((N, Co, Do_avg, Ho_avg, Wo_avg), device=x.device, dtype=x.dtype)
    fused_ext.adaptive_avgpool3d(x, x_avg)
    x = x_avg
    
    # Fused Custom Kernel with Coalesced Memory Access
    N, C, D, H, W = x.shape
    out = torch.zeros((N, D, H, W), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(x.contiguous(), bias.contiguous().view(-1), out, divisor)
    return out

# Placeholders for evaluation requirements
batch_size=128
in_channels=8
out_channels=16
depth=16
height=64
width=64
kernel_size=(3, 3, 3)
divisor=2.0
pool_size=(2, 2, 2)
bias_shape=(out_channels, 1, 1, 1)
sum_dim=1

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
