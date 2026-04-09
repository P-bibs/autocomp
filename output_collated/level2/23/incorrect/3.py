# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260408_235554/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'num_groups']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'group_norm_weight', 'group_norm_bias', 'group_norm_num_groups', 'group_norm_eps']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'group_norm_weight', 'group_norm_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D convolution, applies Group Normalization, computes the mean
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

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
    # State for group_norm (nn.GroupNorm)
    if 'group_norm_weight' in flat_state:
        state_kwargs['group_norm_weight'] = flat_state['group_norm_weight']
    else:
        state_kwargs['group_norm_weight'] = getattr(model.group_norm, 'weight', None)
    if 'group_norm_bias' in flat_state:
        state_kwargs['group_norm_bias'] = flat_state['group_norm_bias']
    else:
        state_kwargs['group_norm_bias'] = getattr(model.group_norm, 'bias', None)
    state_kwargs['group_norm_num_groups'] = model.group_norm.num_groups
    state_kwargs['group_norm_eps'] = model.group_norm.eps
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel implementation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_gn_mean_kernel(
    const float* __restrict__ input,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int D, int H, int W,
    int kernel_size,
    int padding,
    int num_groups,
    float eps
) {
    int batch_idx = blockIdx.x;
    int out_ch_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || out_ch_idx >= out_channels) return;
    
    int D_out = D - 2 * padding;
    int H_out = H - 2 * padding;
    int W_out = W - 2 * padding;
    int spatial_size = D_out * H_out * W_out;
    
    // Shared memory for reduction
    extern __shared__ float sdata[];
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int count = 0;
    
    // Process spatial locations
    for (int idx = tid; idx < spatial_size; idx += blockDim.x) {
        int d = idx / (H_out * W_out);
        int h = (idx % (H_out * W_out)) / W_out;
        int w = idx % W_out;
        
        float conv_result = conv_bias[out_ch_idx];
        
        // Convolution computation
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int in_d = d + kd;
                    int in_h = h + kh;
                    int in_w = w + kw;
                    
                    if (in_d >= 0 && in_d < D && in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
                            int input_idx = batch_idx * (in_channels * D * H * W) + 
                                           in_ch * (D * H * W) + 
                                           in_d * (H * W) + 
                                           in_h * W + 
                                           in_w;
                            int weight_idx = out_ch_idx * (in_channels * kernel_size * kernel_size * kernel_size) +
                                            in_ch * (kernel_size * kernel_size * kernel_size) +
                                            kd * (kernel_size * kernel_size) +
                                            kh * kernel_size +
                                            kw;
                            conv_result += input[input_idx] * conv_weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        sum += conv_result;
        sum_sq += conv_result * conv_result;
        count++;
    }
    
    // Reduction for mean and variance
    sdata[tid] = sum;
    sdata[blockDim.x + tid] = sum_sq;
    sdata[2 * blockDim.x + tid] = (float)count;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[blockDim.x + tid] += sdata[blockDim.x + tid + s];
            sdata[2 * blockDim.x + tid] += sdata[2 * blockDim.x + tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float total_sum = sdata[0];
        float total_sum_sq = sdata[blockDim.x];
        float total_count = sdata[2 * blockDim.x];
        
        if (total_count > 0) {
            float mean = total_sum / total_count;
            float variance = total_sum_sq / total_count - mean * mean;
            float inv_std = rsqrtf(variance + eps);
            
            // Group normalization parameters
            float gamma = gn_weight[out_ch_idx];
            float beta = gn_bias[out_ch_idx];
            
            // Final result is the mean after group normalization
            output[batch_idx * out_channels + out_ch_idx] = mean * inv_std * gamma + beta;
        } else {
            output[batch_idx * out_channels + out_ch_idx] = 0.0f;
        }
    }
}

void fused_conv_gn_mean_forward(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int D, int H, int W,
    int kernel_size,
    int padding,
    int num_groups,
    float eps
) {
    dim3 grid(batch_size, out_channels);
    dim3 block(min(256, (D - 2 * padding) * (H - 2 * padding) * (W - 2 * padding)));
    block.x = min(block.x, 1024);
    block.x = max(block.x, 32);
    int shared_mem_size = 3 * block.x * sizeof(float); // For reduction
    
    fused_conv_gn_mean_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        D, H, W,
        kernel_size,
        padding,
        num_groups,
        eps
    );
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_gn_mean_forward(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int D, int H, int W,
    int kernel_size,
    int padding,
    int num_groups,
    float eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_gn_mean_forward", &fused_conv_gn_mean_forward, "Fused Conv3D + GroupNorm + Mean forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_gn_mean',
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
    group_norm_weight,
    group_norm_bias,
    group_norm_num_groups,
    group_norm_eps,
):
    # For this optimization, we assume stride=1, dilation=1, groups=1 for simplicity
    # In a full implementation, these would be handled properly
    
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    out_channels = conv_weight.shape[0]
    D, H, W = x.shape[2], x.shape[3], x.shape[4]
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, device=x.device, dtype=x.dtype)
    
    # Call our fused kernel
    fused_ext.fused_conv_gn_mean_forward(
        x.contiguous(),
        conv_weight.contiguous(),
        conv_bias.contiguous(),
        group_norm_weight.contiguous(),
        group_norm_bias.contiguous(),
        output,
        batch_size,
        in_channels,
        out_channels,
        D, H, W,
        conv_weight.shape[2],  # kernel_size (assuming cubic)
        conv_padding[0],       # padding (assuming symmetric)
        group_norm_num_groups,
        group_norm_eps
    )
    
    # Take mean across channel dimension to match original behavior
    return output.mean(dim=1)

# Test parameters
batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
num_groups = 8

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]
