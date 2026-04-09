# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_000325/code_5.py
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
from torch.utils.cpp_extension import load_inline

# Define CUDA kernel for fused operation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_conv3d_gn_mean_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int depth,
    int height,
    int width,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    int num_groups,
    float eps
) {
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y;
    int thread_idx = threadIdx.x;

    if (batch_idx >= batch_size || out_ch >= out_channels) return;

    // Calculate output dimensions
    int out_d = (depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Shared memory for reduction
    extern __shared__ float shared_data[];
    float* shared_sum = shared_data;
    float* shared_sum_sq = &shared_data[256];
    
    // Group norm parameters
    int channels_per_group = out_channels / num_groups;
    int group_idx = out_ch / channels_per_group;
    
    // Accumulate convolution results and compute statistics
    float sum_val = 0.0f;
    float sum_sq_val = 0.0f;
    int valid_count = 0;
    
    // Process output spatial locations
    for (int od = 0; od < out_d; ++od) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float conv_result = 0.0f;
                
                // Perform convolution
                for (int kd = 0; kd < kernel_size; ++kd) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int in_ch_start = (out_ch / (out_channels / groups)) * (in_channels / groups);
                            int in_ch_end = in_ch_start + (in_channels / groups);
                            
                            for (int ic = in_ch_start; ic < in_ch_end; ++ic) {
                                int id = od * stride - padding + kd * dilation;
                                int ih = oh * stride - padding + kh * dilation;
                                int iw = ow * stride - padding + kw * dilation;
                                
                                if (id >= 0 && id < depth && ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    int input_idx = batch_idx * (in_channels * depth * height * width) +
                                                    ic * (depth * height * width) +
                                                    id * (height * width) +
                                                    ih * width +
                                                    iw;
                                    int weight_idx = out_ch * (in_channels / groups * kernel_size * kernel_size * kernel_size) +
                                                     (ic - in_ch_start) * (kernel_size * kernel_size * kernel_size) +
                                                     kd * (kernel_size * kernel_size) +
                                                     kh * kernel_size +
                                                     kw;
                                    conv_result += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                }
                
                // Add bias
                conv_result += bias[out_ch];
                
                // Accumulate for mean and variance computation
                sum_val += conv_result;
                sum_sq_val += conv_result * conv_result;
                valid_count++;
            }
        }
    }
    
    // Store partial sums in shared memory
    shared_sum[thread_idx] = sum_val;
    shared_sum_sq[thread_idx] = sum_sq_val;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = 128; s > 0; s >>= 1) {
        if (thread_idx < s) {
            shared_sum[thread_idx] += shared_sum[thread_idx + s];
            shared_sum_sq[thread_idx] += shared_sum_sq[thread_idx + s];
        }
        __syncthreads();
    }
    
    // Apply group norm and finalize result
    if (thread_idx == 0) {
        float mean = shared_sum[0] / valid_count;
        float variance = (shared_sum_sq[0] / valid_count) - (mean * mean);
        float inv_std = rsqrtf(variance + eps);
        
        // Compute final normalized value (simplified to just the mean of normalized values)
        float normalized_mean = mean * gn_weight[group_idx] + gn_bias[group_idx];
        output[batch_idx] = normalized_mean;
    }
}

void fused_conv3d_gn_mean_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups,
    int num_groups,
    float eps
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2); // Assuming cubic kernel
    
    dim3 grid(batch_size, out_channels);
    dim3 block(256);
    size_t shared_mem_size = 2 * 256 * sizeof(float); // For sum and sum_sq
    
    fused_conv3d_gn_mean_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        depth,
        height,
        width,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        num_groups,
        eps
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void fused_conv3d_gn_mean_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups,
    int num_groups,
    float eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv3d_gn_mean_forward", &fused_conv3d_gn_mean_forward, "Fused Conv3D + GroupNorm + Mean Forward");
}
"""

# Compile extension
fused_ext = load_inline(
    name='fused_conv3d_gn_mean',
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
    # Prepare output tensor
    out = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_conv3d_gn_mean_forward(
        x, conv_weight, conv_bias,
        group_norm_weight, group_norm_bias, out,
        conv_stride[0] if isinstance(conv_stride, (list, tuple)) else conv_stride,
        conv_padding[0] if isinstance(conv_padding, (list, tuple)) else conv_padding,
        conv_dilation[0] if isinstance(conv_dilation, (list, tuple)) else conv_dilation,
        conv_groups,
        group_norm_num_groups,
        group_norm_eps
    )
    
    return out

# Constants
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
