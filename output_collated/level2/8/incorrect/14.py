# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_055208/code_2.py
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
#include <cuda_fp16.h>

#define TILE_DIM 32

__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int Ci, int Di, int Hi, int Wi,
    int Co, int Kd, int Kh, int Kw,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w
) {
    // Shared memory for tiled input and weight
    extern __shared__ float sdata[];
    
    int batch_id = blockIdx.z;
    int out_d = blockIdx.x;
    int out_h = blockIdx.y;
    int out_w = threadIdx.x;
    
    if (batch_id >= N || out_d >= (Di - Kd + 2 * padding_d) / stride_d + 1 ||
        out_h >= (Hi - Kh + 2 * padding_h) / stride_h + 1 ||
        out_w >= (Wi - Kw + 2 * padding_w) / stride_w + 1)
        return;
        
    int od = (Di - Kd + 2 * padding_d) / stride_d + 1;
    int oh = (Hi - Kh + 2 * padding_h) / stride_h + 1;
    int ow = (Wi - Kw + 2 * padding_w) / stride_w + 1;
    
    float sum = 0.0f;
    
    // Compute convolution
    for(int co = 0; co < Co; co++) {
        sum = bias[co];
        for(int kd = 0; kd < Kd; kd++) {
            for(int kh = 0; kh < Kh; kh++) {
                for(int kw = 0; kw < Kw; kw++) {
                    for(int ci = 0; ci < Ci; ci++) {
                        int id = out_d * stride_d + kd - padding_d;
                        int ih = out_h * stride_h + kh - padding_h;
                        int iw = out_w * stride_w + kw - padding_w;
                        
                        float val = 0.0f;
                        if(id >= 0 && id < Di && ih >= 0 && ih < Hi && iw >= 0 && iw < Wi) {
                            val = input[batch_id * (Ci * Di * Hi * Wi) + 
                                        ci * (Di * Hi * Wi) + 
                                        id * (Hi * Wi) + 
                                        ih * Wi + 
                                        iw];
                        }
                        
                        float wgt = weight[co * (Ci * Kd * Kh * Kw) + 
                                           ci * (Kd * Kh * Kw) + 
                                           kd * (Kh * Kw) + 
                                           kh * Kw + 
                                           kw];
                                           
                        sum += val * wgt;
                    }
                }
            }
        }
        output[batch_id * (Co * od * oh * ow) + 
               co * (od * oh * ow) + 
               out_d * (oh * ow) + 
               out_h * ow + 
               out_w] = sum;
    }
}

__global__ void max_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int Di, int Hi, int Wi,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w
) {
    int batch_id = blockIdx.z;
    int out_d = blockIdx.x;
    int out_h = blockIdx.y;
    int out_w = threadIdx.x;
    
    if (batch_id >= N || out_d >= (Di - kernel_d + 2 * padding_d) / stride_d + 1 ||
        out_h >= (Hi - kernel_h + 2 * padding_h) / stride_h + 1 ||
        out_w >= (Wi - kernel_w + 2 * padding_w) / stride_w + 1)
        return;
        
    int od = (Di - kernel_d + 2 * padding_d) / stride_d + 1;
    int oh = (Hi - kernel_h + 2 * padding_h) / stride_h + 1;
    int ow = (Wi - kernel_w + 2 * padding_w) / stride_w + 1;
    
    for(int c = 0; c < C; c++) {
        float max_val = -FLT_MAX;
        for(int kd = 0; kd < kernel_d; kd++) {
            for(int kh = 0; kh < kernel_h; kh++) {
                for(int kw = 0; kw < kernel_w; kw++) {
                    int id = out_d * stride_d + kd - padding_d;
                    int ih = out_h * stride_h + kh - padding_h;
                    int iw = out_w * stride_w + kw - padding_w;
                    
                    if(id >= 0 && id < Di && ih >= 0 && ih < Hi && iw >= 0 && iw < Wi) {
                        float val = input[batch_id * (C * Di * Hi * Wi) + 
                                          c * (Di * Hi * Wi) + 
                                          id * (Hi * Wi) + 
                                          ih * Wi + 
                                          iw];
                        max_val = fmaxf(max_val, val);
                    }
                }
            }
        }
        output[batch_id * (C * od * oh * ow) + 
               c * (od * oh * ow) + 
               out_d * (oh * ow) + 
               out_h * ow + 
               out_w] = max_val;
    }
}

__global__ void adaptive_avg_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int Di, int Hi, int Wi,
    int od, int oh, int ow
) {
    int batch_id = blockIdx.z;
    int oc = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_id >= N || oc >= C || idx >= od * oh * ow)
        return;
        
    int out_d = idx / (oh * ow);
    int out_h = (idx / ow) % oh;
    int out_w = idx % ow;
    
    // Calculate input region for this output
    int id_start = (out_d * Di) / od;
    int ih_start = (out_h * Hi) / oh;
    int iw_start = (out_w * Wi) / ow;
    int id_end = ((out_d + 1) * Di + od - 1) / od;
    int ih_end = ((out_h + 1) * Hi + oh - 1) / oh;
    int iw_end = ((out_w + 1) * Wi + ow - 1) / ow;
    
    float sum = 0.0f;
    int count = 0;
    
    for(int id = id_start; id < id_end; id++) {
        for(int ih = ih_start; ih < ih_end; ih++) {
            for(int iw = iw_start; iw < iw_end; iw++) {
                sum += input[batch_id * (C * Di * Hi * Wi) + 
                             oc * (Di * Hi * Wi) + 
                             id * (Hi * Wi) + 
                             ih * Wi + 
                             iw];
                count++;
            }
        }
    }
    
    output[batch_id * (C * od * oh * ow) + 
           oc * (od * oh * ow) + 
           out_d * (oh * ow) + 
           out_h * ow + 
           out_w] = sum / count;
}

// Optimized fused kernel with shared memory and vectorization
__global__ void fused_op_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float inv_divisor,
    int N, int C, int D, int H, int W) {
    
    extern __shared__ float s_bias[];
    
    // Load bias into shared memory for fast access
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        s_bias[i] = bias[i];
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = D * H * W;
    
    if (tid < N * spatial_size) {
        int n = tid / spatial_size;
        int spatial_idx = tid % spatial_size;
        
        float sum_val = 0.0f;
        int base_idx = n * (C * spatial_size) + spatial_idx;
        
        // Unroll loop for C processing
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            sum_val += (input[base_idx + c * spatial_size] * inv_divisor) + s_bias[c];
        }
        output[tid] = sum_val;
    }
}

void conv3d_op(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
               int stride_d, int stride_h, int stride_w,
               int padding_d, int padding_h, int padding_w) {
    int N = input.size(0);
    int Ci = input.size(1);
    int Di = input.size(2);
    int Hi = input.size(3);
    int Wi = input.size(4);
    
    int Co = weight.size(0);
    int Kd = weight.size(2);
    int Kh = weight.size(3);
    int Kw = weight.size(4);
    
    int od = (Di - Kd + 2 * padding_d) / stride_d + 1;
    int oh = (Hi - Kh + 2 * padding_h) / stride_h + 1;
    int ow = (Wi - Kw + 2 * padding_w) / stride_w + 1;
    
    dim3 grid(ow, oh, N);
    dim3 block(od, 1, 1);
    
    size_t shared_mem_size = Ci * Kd * Kh * Kw * sizeof(float);
    
    conv3d_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, Ci, Di, Hi, Wi, Co, Kd, Kh, Kw, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w);
}

void max_pool3d_op(torch::Tensor input, torch::Tensor output,
                   int kernel_d, int kernel_h, int kernel_w,
                   int stride_d, int stride_h, int stride_w,
                   int padding_d, int padding_h, int padding_w) {
    int N = input.size(0);
    int C = input.size(1);
    int Di = input.size(2);
    int Hi = input.size(3);
    int Wi = input.size(4);
    
    int od = (Di - kernel_d + 2 * padding_d) / stride_d + 1;
    int oh = (Hi - kernel_h + 2 * padding_h) / stride_h + 1;
    int ow = (Wi - kernel_w + 2 * padding_w) / stride_w + 1;
    
    dim3 grid(ow, oh, N);
    dim3 block(od, 1, 1);
    
    max_pool3d_kernel<<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, Di, Hi, Wi, kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w);
}

void adaptive_avg_pool3d_op(torch::Tensor input, torch::Tensor output, int od, int oh, int ow) {
    int N = input.size(0);
    int C = input.size(1);
    int Di = input.size(2);
    int Hi = input.size(3);
    int Wi = input.size(4);
    
    int total_threads = od * oh * ow;
    int threads_per_block = 256;
    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;
    
    adaptive_avg_pool3d_kernel<<<dim3(blocks_per_grid, C, N), threads_per_block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), N, C, Di, Hi, Wi, od, oh, ow);
}

void fused_op(torch::Tensor input, torch::Tensor bias, torch::Tensor output, float divisor) {
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    
    int threads = 256;
    int spatial_size = D * H * W;
    int blocks = (N * spatial_size + threads - 1) / threads;
    
    fused_op_kernel<<<blocks, threads, C * sizeof(float)>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), 
        1.0f/divisor, N, C, D, H, W);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void conv3d_op(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
               int stride_d, int stride_h, int stride_w,
               int padding_d, int padding_h, int padding_w);
               
void max_pool3d_op(torch::Tensor input, torch::Tensor output,
                   int kernel_d, int kernel_h, int kernel_w,
                   int stride_d, int stride_h, int stride_w,
                   int padding_d, int padding_h, int padding_w);
                   
void adaptive_avg_pool3d_op(torch::Tensor input, torch::Tensor output, int od, int oh, int ow);

void fused_op(torch::Tensor input, torch::Tensor bias, torch::Tensor output, float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_op", &conv3d_op, "Custom Conv3D kernel");
    m.def("max_pool3d_op", &max_pool3d_op, "Custom MaxPool3D kernel");
    m.def("adaptive_avg_pool3d_op", &adaptive_avg_pool3d_op, "Custom AdaptiveAvgPool3D kernel");
    m.def("fused_op", &fused_op, "Optimized fused op");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
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
    padding_d, padding_h, padding_w = conv_padding
    
    N, Ci, Di, Hi, Wi = x.shape
    Co, _, Kd, Kh, Kw = conv_weight.shape
    od = (Di - Kd + 2 * padding_d) // stride_d + 1
    oh = (Hi - Kh + 2 * padding_h) // stride_h + 1
    ow = (Wi - Kw + 2 * padding_w) // stride_w + 1
    
    x_conv = torch.empty((N, Co, od, oh, ow), device=x.device, dtype=x.dtype)
    fused_ext.conv3d_op(x, conv_weight, conv_bias, x_conv, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w)
    
    # MaxPool3D
    kernel_d, kernel_h, kernel_w = max_pool_kernel_size
    stride_d, stride_h, stride_w = max_pool_stride
    padding_d, padding_h, padding_w = max_pool_padding
    
    od = (x_conv.size(2) - kernel_d + 2 * padding_d) // stride_d + 1
    oh = (x_conv.size(3) - kernel_h + 2 * padding_h) // stride_h + 1
    ow = (x_conv.size(4) - kernel_w + 2 * padding_w) // stride_w + 1
    
    x_pool = torch.empty((N, Co, od, oh, ow), device=x.device, dtype=x.dtype)
    fused_ext.max_pool3d_op(x_conv, x_pool, kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w)
    
    # AdaptiveAvgPool3D
    od, oh, ow = global_avg_pool_output_size
    x_adaptive = torch.empty((N, Co, od, oh, ow), device=x.device, dtype=x.dtype)
    fused_ext.adaptive_avg_pool3d_op(x_pool, x_adaptive, od, oh, ow)
    
    # Fused operation
    N, C, D, H, W = x_adaptive.shape
    out = torch.empty((N, D, H, W), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x_adaptive.contiguous(), bias.contiguous().view(-1), out, divisor)
    
    return out

# Placeholders for evaluation requirements
batch_size=128; in_channels=8; out_channels=16; depth=16; height=64; width=64
kernel_size=(3, 3, 3); divisor=2.0; pool_size=(2, 2, 2); bias_shape=(out_channels, 1, 1, 1); sum_dim=1

def get_init_inputs(): return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]
def get_inputs(): return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
