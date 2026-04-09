# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_100332/code_0.py
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

# Fused CUDA kernel implementation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define MAX_CHANNELS 64

__global__ void fused_conv_min_softmax_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int D, int H, int W,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups,
    int dim
) {
    // Calculate output dimensions
    int outD = (D + 2 * pad_d - dilation_d * (kD - 1) - 1) / stride_d + 1;
    int outH = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int outW = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * outD * outH * outW;
    
    if (tid >= total_output_elements) return;
    
    // Decode output position
    int tmp = tid;
    int w_out = tmp % outW; tmp /= outW;
    int h_out = tmp % outH; tmp /= outH;
    int d_out = tmp % outD; tmp /= outD;
    int c_out = tmp % out_channels; tmp /= out_channels;
    int b = tmp;
    
    // Calculate convolution
    float conv_result = 0.0f;
    int group_idx = c_out / (out_channels / groups);
    int channels_per_group = in_channels / groups;
    
    for (int kc = 0; kc < channels_per_group; kc++) {
        for (int kd = 0; kd < kD; kd++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    int c_in = group_idx * channels_per_group + kc;
                    int d_in = d_out * stride_d - pad_d + kd * dilation_d;
                    int h_in = h_out * stride_h - pad_h + kh * dilation_h;
                    int w_in = w_out * stride_w - pad_w + kw * dilation_w;
                    
                    if (d_in >= 0 && d_in < D && h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        int input_idx = ((b * in_channels + c_in) * D + d_in) * H * W + h_in * W + w_in;
                        int weight_idx = ((c_out * channels_per_group + kc) * kD + kd) * kH * kW + kh * kW + kw;
                        conv_result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    conv_result += bias[c_out];
    output[tid] = conv_result;
}

// Fully fused kernel that includes min reduction and softmax
__global__ void fully_fused_conv_min_softmax_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int D, int H, int W,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups,
    int dim
) {
    // Calculate output dimensions
    int outD = (D + 2 * pad_d - dilation_d * (kD - 1) - 1) / stride_d + 1;
    int outH = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int outW = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1;
    
    // For min reduction along dim=2 (depth), we need to handle different output dimensions
    int reduced_outD = (dim == 2) ? 1 : outD;
    int reduced_outH = (dim == 3) ? 1 : outH;
    int reduced_outW = (dim == 4) ? 1 : outW;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_reduced_elements = batch_size * out_channels * reduced_outD * reduced_outH * reduced_outW;
    
    if (tid >= total_reduced_elements) return;
    
    // Decode reduced output position
    int tmp = tid;
    int w_idx = tmp % reduced_outW; tmp /= reduced_outW;
    int h_idx = tmp % reduced_outH; tmp /= reduced_outH;
    int d_idx = tmp % reduced_outD; tmp /= reduced_outD;
    int c_out = tmp % out_channels; tmp /= out_channels;
    int b = tmp;
    
    // Apply min reduction along specified dimension
    float min_val = INFINITY;
    
    if (dim == 2) { // Reduce along depth (D)
        for (int d = 0; d < outD; d++) {
            // Calculate convolution for this position
            float conv_result = 0.0f;
            int group_idx = c_out / (out_channels / groups);
            int channels_per_group = in_channels / groups;
            
            for (int kc = 0; kc < channels_per_group; kc++) {
                for (int kd = 0; kd < kD; kd++) {
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int c_in = group_idx * channels_per_group + kc;
                            int d_in = d * stride_d - pad_d + kd * dilation_d;
                            int h_in = h_idx * stride_h - pad_h + kh * dilation_h;
                            int w_in = w_idx * stride_w - pad_w + kw * dilation_w;
                            
                            if (d_in >= 0 && d_in < D && h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                int input_idx = ((b * in_channels + c_in) * D + d_in) * H * W + h_in * W + w_in;
                                int weight_idx = ((c_out * channels_per_group + kc) * kD + kd) * kH * kW + kh * kW + kw;
                                conv_result += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
            
            conv_result += bias[c_out];
            min_val = fminf(min_val, conv_result);
        }
    } else if (dim == 3) { // Reduce along height (H)
        for (int h = 0; h < outH; h++) {
            // Calculate convolution for this position
            float conv_result = 0.0f;
            int group_idx = c_out / (out_channels / groups);
            int channels_per_group = in_channels / groups;
            
            for (int kc = 0; kc < channels_per_group; kc++) {
                for (int kd = 0; kd < kD; kd++) {
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int c_in = group_idx * channels_per_group + kc;
                            int d_in = d_idx * stride_d - pad_d + kd * dilation_d;
                            int h_in = h * stride_h - pad_h + kh * dilation_h;
                            int w_in = w_idx * stride_w - pad_w + kw * dilation_w;
                            
                            if (d_in >= 0 && d_in < D && h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                int input_idx = ((b * in_channels + c_in) * D + d_in) * H * W + h_in * W + w_in;
                                int weight_idx = ((c_out * channels_per_group + kc) * kD + kd) * kH * kW + kh * kW + kw;
                                conv_result += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
            
            conv_result += bias[c_out];
            min_val = fminf(min_val, conv_result);
        }
    } else if (dim == 4) { // Reduce along width (W)
        for (int w = 0; w < outW; w++) {
            // Calculate convolution for this position
            float conv_result = 0.0f;
            int group_idx = c_out / (out_channels / groups);
            int channels_per_group = in_channels / groups;
            
            for (int kc = 0; kc < channels_per_group; kc++) {
                for (int kd = 0; kd < kD; kd++) {
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int c_in = group_idx * channels_per_group + kc;
                            int d_in = d_idx * stride_d - pad_d + kd * dilation_d;
                            int h_in = h_idx * stride_h - pad_h + kh * dilation_h;
                            int w_in = w * stride_w - pad_w + kw * dilation_w;
                            
                            if (d_in >= 0 && d_in < D && h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                int input_idx = ((b * in_channels + c_in) * D + d_in) * H * W + h_in * W + w_in;
                                int weight_idx = ((c_out * channels_per_group + kc) * kD + kd) * kH * kW + kh * kW + kw;
                                conv_result += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
            
            conv_result += bias[c_out];
            min_val = fminf(min_val, conv_result);
        }
    }
    
    // Store min result
    output[tid] = min_val;
}

// Kernel for softmax along dim=1
__global__ void softmax_kernel(
    float* __restrict__ input_output,
    int batch_size,
    int channels,
    int spatial_size
) {
    int batch_idx = blockIdx.x;
    int spatial_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || spatial_idx >= spatial_size) return;
    
    // Shared memory for reduction
    extern __shared__ float shared_data[];
    float* shared_vals = shared_data;
    
    // Load data into shared memory
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        int idx = batch_idx * channels * spatial_size + c * spatial_size + spatial_idx;
        shared_vals[c] = input_output[idx];
    }
    __syncthreads();
    
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int c = 0; c < channels; c++) {
        max_val = fmaxf(max_val, shared_vals[c]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int c = 0; c < channels; c++) {
        shared_vals[c] = expf(shared_vals[c] - max_val);
        sum += shared_vals[c];
    }
    
    // Normalize
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        shared_vals[c] /= sum;
        int idx = batch_idx * channels * spatial_size + c * spatial_size + spatial_idx;
        input_output[idx] = shared_vals[c];
    }
}

void fused_conv_min_softmax_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups,
    int dim
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    
    auto out_channels = weight.size(0);
    auto kD = weight.size(2);
    auto kH = weight.size(3);
    auto kW = weight.size(4);
    
    auto outD = (D + 2 * pad_d - dilation_d * (kD - 1) - 1) / stride_d + 1;
    auto outH = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    auto outW = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1;
    
    // First, perform fused conv + min
    int reduced_outD = (dim == 2) ? 1 : outD;
    int reduced_outH = (dim == 3) ? 1 : outH;
    int reduced_outW = (dim == 4) ? 1 : outW;
    
    int total_reduced_elements = batch_size * out_channels * reduced_outD * reduced_outH * reduced_outW;
    int threads_per_block = 256;
    int blocks = (total_reduced_elements + threads_per_block - 1) / threads_per_block;
    
    fully_fused_conv_min_softmax_kernel<<<blocks, threads_per_block>>>(
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
        dilation_d, dilation_h, dilation_w,
        groups,
        dim
    );
}

void softmax_forward(
    torch::Tensor& input_output,
    int batch_size,
    int channels,
    int spatial_size
) {
    dim3 block_size(256);
    dim3 grid_size(batch_size, (spatial_size + block_size.x - 1) / block_size.x);
    size_t shared_mem_size = channels * sizeof(float);
    
    softmax_kernel<<<grid_size, block_size, shared_mem_size>>>(
        input_output.data_ptr<float>(),
        batch_size,
        channels,
        spatial_size
    );
}
"""

# C++ interface bindings
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_min_softmax_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups,
    int dim
);

void softmax_forward(
    torch::Tensor& input_output,
    int batch_size,
    int channels,
    int spatial_size
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_softmax", &fused_conv_min_softmax_forward, "Fused conv3d + min operation");
    m.def("softmax", &softmax_forward, "Softmax operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
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
    # Extract convolution parameters
    if isinstance(conv_stride, int):
        stride_d = stride_h = stride_w = conv_stride
    else:
        stride_d, stride_h, stride_w = conv_stride
    
    if isinstance(conv_padding, int):
        pad_d = pad_h = pad_w = conv_padding
    else:
        pad_d, pad_h, pad_w = conv_padding
    
    if isinstance(conv_dilation, int):
        dilation_d = dilation_h = dilation_w = conv_dilation
    else:
        dilation_d, dilation_h, dilation_w = conv_dilation
    
    # Calculate output dimensions
    batch_size, in_channels, D, H, W = x.shape
    out_channels = conv_weight.shape[0]
    kD, kH, kW = conv_weight.shape[2], conv_weight.shape[3], conv_weight.shape[4]
    
    outD = (D + 2 * pad_d - dilation_d * (kD - 1) - 1) // stride_d + 1
    outH = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) // stride_h + 1
    outW = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) // stride_w + 1
    
    # Adjust for min reduction along specified dimension
    if dim == 2:  # Reduce along depth
        reduced_D = 1
        reduced_H = outH
        reduced_W = outW
    elif dim == 3:  # Reduce along height
        reduced_D = outD
        reduced_H = 1
        reduced_W = outW
    elif dim == 4:  # Reduce along width
        reduced_D = outD
        reduced_H = outH
        reduced_W = 1
    else:
        raise ValueError("dim must be 2, 3, or 4 for depth, height, or width respectively")
    
    # Create output tensor with reduced dimensions
    output = torch.empty(batch_size, out_channels, reduced_D, reduced_H, reduced_W, device=x.device, dtype=x.dtype)
    
    # Call fused conv + min operation
    fused_ext.fused_conv_min_softmax(
        x, conv_weight, conv_bias, output,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w,
        conv_groups, dim
    )
    
    # Apply softmax along channel dimension (dim=1)
    spatial_size = output.shape[2] * output.shape[3] * output.shape[4]
    fused_ext.softmax(output, batch_size, out_channels, spatial_size)
    
    return output

# Constants (same as original)
batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
dim = 2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]
