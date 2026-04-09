# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_100757/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a 3D convolution, applies minimum operation along a specific dimension, 
    and then applies softmax.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

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
    if 'dim' in flat_state:
        state_kwargs['dim'] = flat_state['dim']
    else:
        state_kwargs['dim'] = getattr(model, 'dim')
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

# CUDA kernel that fuses conv3d + min + softmax
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

__global__ void fused_conv_min_softmax_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int D, int H, int W,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w) {
    
    // Calculate output dimensions
    int out_d = (D + 2 * pad_d - dilation_d * (kD - 1) - 1) / stride_d + 1;
    int out_h = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int out_w = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1;
    
    int spatial_size = out_h * out_w;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= batch_size * spatial_size) return;
    
    // Decode thread index
    int w_idx = tid % out_w;
    int h_idx = (tid / out_w) % out_h;
    int b_idx = tid / spatial_size;
    
    // Shared memory for softmax computation
    extern __shared__ float shared_mem[];
    float* softmax_vals = shared_mem;
    
    // For each output channel, compute min across depth
    for (int ch = 0; ch < out_channels; ch++) {
        float min_val = INFINITY;
        
        // Compute min across depth dimension
        for (int d_idx = 0; d_idx < out_d; d_idx++) {
            float conv_sum = 0.0f;
            
            // Perform convolution at this position
            for (int kd = 0; kd < kD; kd++) {
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        for (int ic = 0; ic < in_channels; ic++) {
                            int in_d = d_idx * stride_d - pad_d + kd * dilation_d;
                            int in_h = h_idx * stride_h - pad_h + kh * dilation_h;
                            int in_w = w_idx * stride_w - pad_w + kw * dilation_w;
                            
                            if (in_d >= 0 && in_d < D && in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                                int input_idx = b_idx * (in_channels * D * H * W) + 
                                               ic * (D * H * W) + 
                                               in_d * (H * W) + 
                                               in_h * W + in_w;
                                               
                                int weight_idx = ch * (in_channels * kD * kH * kW) + 
                                                ic * (kD * kH * kW) + 
                                                kd * (kH * kW) + 
                                                kh * kW + kw;
                                                
                                conv_sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
            
            // Add bias
            conv_sum += bias[ch];
            min_val = fminf(min_val, conv_sum);
        }
        
        // Store min value for softmax calculation
        softmax_vals[threadIdx.x * out_channels + ch] = min_val;
    }
    
    // Synchronize threads in block
    __syncthreads();
    
    // Compute softmax across channels
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int ch = 0; ch < out_channels; ch++) {
        max_val = fmaxf(max_val, softmax_vals[threadIdx.x * out_channels + ch]);
    }
    
    // Compute exponentials and sum
    float sum_exp = 0.0f;
    for (int ch = 0; ch < out_channels; ch++) {
        float exp_val = expf(softmax_vals[threadIdx.x * out_channels + ch] - max_val);
        softmax_vals[threadIdx.x * out_channels + ch] = exp_val;
        sum_exp += exp_val;
    }
    
    // Normalize and write output
    for (int ch = 0; ch < out_channels; ch++) {
        int output_idx = b_idx * (out_channels * out_h * out_w) +
                        ch * (out_h * out_w) +
                        h_idx * out_w + w_idx;
        output[output_idx] = softmax_vals[threadIdx.x * out_channels + ch] / sum_exp;
    }
}

void launch_fused_conv_min_softmax(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int D, int H, int W,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w) {
    
    int out_h = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int out_w = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1;
    int spatial_size = out_h * out_w;
    int total_threads = batch_size * spatial_size;
    
    int threads_per_block = min(256, spatial_size);
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    size_t shared_mem_size = threads_per_block * out_channels * sizeof(float);
    
    fused_conv_min_softmax_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        input, weight, bias, output,
        batch_size, in_channels, out_channels,
        D, H, W, kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w
    );
}
"""

# C++ interface bindings
cpp_source = r"""
#include <torch/extension.h>

void launch_fused_conv_min_softmax(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int D, int H, int W,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w);

void fused_conv_min_softmax(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int D, int H, int W,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w) {
    
    launch_fused_conv_min_softmax(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        D, H, W,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_softmax", &fused_conv_min_softmax, "Fused Conv3D + Min + Softmax");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_min_softmax_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    dim,
):
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    D, H, W = x.shape[2], x.shape[3], x.shape[4]
    out_channels = conv_weight.shape[0]
    kD, kH, kW = conv_weight.shape[2], conv_weight.shape[3], conv_weight.shape[4]
    
    # Handle tuple or single values for stride, padding, dilation
    def _to_tuple(val, n):
        if isinstance(val, (tuple, list)):
            return tuple(val)
        return tuple([val] * n)
    
    stride_d, stride_h, stride_w = _to_tuple(conv_stride, 3)
    pad_d, pad_h, pad_w = _to_tuple(conv_padding, 3)
    dilation_d, dilation_h, dilation_w = _to_tuple(conv_dilation, 3)
    
    # Calculate output dimensions
    out_h = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) // stride_h + 1
    out_w = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) // stride_w + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)
    
    # Launch fused kernel
    fused_ext.fused_conv_min_softmax(
        x, conv_weight, conv_bias, output,
        batch_size, in_channels, out_channels,
        D, H, W, kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w
    )
    
    return output

# Test parameters
batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
dim = 2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W, device='cuda').requires_grad_()]
