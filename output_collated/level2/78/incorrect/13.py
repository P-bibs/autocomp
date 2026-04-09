# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_031947/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'max_pool1_kernel_size', 'max_pool1_stride', 'max_pool1_padding', 'max_pool1_dilation', 'max_pool1_ceil_mode', 'max_pool1_return_indices', 'max_pool2_kernel_size', 'max_pool2_stride', 'max_pool2_padding', 'max_pool2_dilation', 'max_pool2_ceil_mode', 'max_pool2_return_indices']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

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
    # State for conv_transpose (nn.ConvTranspose3d)
    if 'conv_transpose_weight' in flat_state:
        state_kwargs['conv_transpose_weight'] = flat_state['conv_transpose_weight']
    else:
        state_kwargs['conv_transpose_weight'] = getattr(model.conv_transpose, 'weight', None)
    if 'conv_transpose_bias' in flat_state:
        state_kwargs['conv_transpose_bias'] = flat_state['conv_transpose_bias']
    else:
        state_kwargs['conv_transpose_bias'] = getattr(model.conv_transpose, 'bias', None)
    state_kwargs['conv_transpose_stride'] = model.conv_transpose.stride
    state_kwargs['conv_transpose_padding'] = model.conv_transpose.padding
    state_kwargs['conv_transpose_output_padding'] = model.conv_transpose.output_padding
    state_kwargs['conv_transpose_groups'] = model.conv_transpose.groups
    state_kwargs['conv_transpose_dilation'] = model.conv_transpose.dilation
    # State for max_pool1 (nn.MaxPool3d)
    state_kwargs['max_pool1_kernel_size'] = model.max_pool1.kernel_size
    state_kwargs['max_pool1_stride'] = model.max_pool1.stride
    state_kwargs['max_pool1_padding'] = model.max_pool1.padding
    state_kwargs['max_pool1_dilation'] = model.max_pool1.dilation
    state_kwargs['max_pool1_ceil_mode'] = model.max_pool1.ceil_mode
    state_kwargs['max_pool1_return_indices'] = model.max_pool1.return_indices
    # State for max_pool2 (nn.MaxPool3d)
    state_kwargs['max_pool2_kernel_size'] = model.max_pool2.kernel_size
    state_kwargs['max_pool2_stride'] = model.max_pool2.stride
    state_kwargs['max_pool2_padding'] = model.max_pool2.padding
    state_kwargs['max_pool2_dilation'] = model.max_pool2.dilation
    state_kwargs['max_pool2_ceil_mode'] = model.max_pool2.ceil_mode
    state_kwargs['max_pool2_return_indices'] = model.max_pool2.return_indices
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

__global__ void fused_conv_transpose_pool3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth, 
    int input_height,
    int input_width,
    int output_depth,
    int output_height, 
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int pool_kernel_size,
    int pool_stride,
    int pool_padding) {
    
    // Calculate global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    CUDA_1D_KERNEL_LOOP(index, total_elements) {
        // Decode output tensor coordinates
        int w = index % output_width;
        int h = (index / output_width) % output_height;
        int d = (index / (output_width * output_height)) % output_depth;
        int c = (index / (output_width * output_height * output_depth)) % out_channels;
        int b = index / (output_width * output_height * output_depth * out_channels);
        
        if (b >= batch_size) return;
        
        // ConvTranspose3d computation
        float conv_result = (bias != nullptr) ? bias[c] : 0.0f;
        
        // Determine convolution input range
        int start_kd = max(0, (d + padding - kernel_size + 1 + stride - 1) / stride);
        int end_kd = min(input_depth, (d + padding + stride) / stride);
        int start_kh = max(0, (h + padding - kernel_size + 1 + stride - 1) / stride);
        int end_kh = min(input_height, (h + padding + stride) / stride);
        int start_kw = max(0, (w + padding - kernel_size + 1 + stride - 1) / stride);
        int end_kw = min(input_width, (w + padding + stride) / stride);
        
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = start_kd; kd < end_kd; ++kd) {
                for (int kh = start_kh; kh < end_kh; ++kh) {
                    for (int kw = start_kw; kw < end_kw; ++kw) {
                        int wd = d + padding - kd * stride;
                        int wh = h + padding - kh * stride;
                        int ww = w + padding - kw * stride;
                        
                        if (wd >= 0 && wd < kernel_size && 
                            wh >= 0 && wh < kernel_size && 
                            ww >= 0 && ww < kernel_size) {
                            
                            int input_idx = ((((b * in_channels) + ic) * input_depth + kd) * input_height + kh) * input_width + kw;
                            int weight_idx = ((((c * in_channels) + ic) * kernel_size + wd) * kernel_size + wh) * kernel_size + ww;
                            
                            conv_result += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // MaxPool3d computation (pool along spatial dimensions)
        float pooled_result = conv_result;
        
        // Determine pooling window boundaries
        int pool_start_d = max(0, d - (pool_kernel_size - 1) * pool_stride / 2);
        int pool_end_d = min(output_depth - 1, d + (pool_kernel_size - 1) * pool_stride / 2);
        int pool_start_h = max(0, h - (pool_kernel_size - 1) * pool_stride / 2);
        int pool_end_h = min(output_height - 1, h + (pool_kernel_size - 1) * pool_stride / 2);
        int pool_start_w = max(0, w - (pool_kernel_size - 1) * pool_stride / 2);
        int pool_end_w = min(output_width - 1, w + (pool_kernel_size - 1) * pool_stride / 2);
        
        // Adjust to align with pooling grid
        pool_start_d = ((pool_start_d + pool_padding) / pool_stride) * pool_stride - pool_padding;
        pool_start_h = ((pool_start_h + pool_padding) / pool_stride) * pool_stride - pool_padding;
        pool_start_w = ((pool_start_w + pool_padding) / pool_stride) * pool_stride - pool_padding;
        
        for (int pd = pool_start_d; pd <= pool_end_d; pd += pool_stride) {
            for (int ph = pool_start_h; ph <= pool_end_h; ph += pool_stride) {
                for (int pw = pool_start_w; pw <= pool_end_w; pw += pool_stride) {
                    if (pd < 0 || ph < 0 || pw < 0 || 
                        pd >= output_depth || ph >= output_height || pw >= output_width) continue;
                    
                    if (pd == d && ph == h && pw == w) continue; // Skip self
                    
                    // Recompute convolution at pooled positions
                    float neighbor_conv = (bias != nullptr) ? bias[c] : 0.0f;
                    
                    // Same conv computation as above
                    int n_start_kd = max(0, (pd + padding - kernel_size + 1 + stride - 1) / stride);
                    int n_end_kd = min(input_depth, (pd + padding + stride) / stride);
                    int n_start_kh = max(0, (ph + padding - kernel_size + 1 + stride - 1) / stride);
                    int n_end_kh = min(input_height, (ph + padding + stride) / stride);
                    int n_start_kw = max(0, (pw + padding - kernel_size + 1 + stride - 1) / stride);
                    int n_end_kw = min(input_width, (pw + padding + stride) / stride);
                    
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kd = n_start_kd; kd < n_end_kd; ++kd) {
                            for (int kh = n_start_kh; kh < n_end_kh; ++kh) {
                                for (int kw = n_start_kw; kw < n_end_kw; ++kw) {
                                    int wd = pd + padding - kd * stride;
                                    int wh = ph + padding - kh * stride;
                                    int ww = pw + padding - kw * stride;
                                    
                                    if (wd >= 0 && wd < kernel_size && 
                                        wh >= 0 && wh < kernel_size && 
                                        ww >= 0 && ww < kernel_size) {
                                        
                                        int input_idx = ((((b * in_channels) + ic) * input_depth + kd) * input_height + kh) * input_width + kw;
                                        int weight_idx = ((((c * in_channels) + ic) * kernel_size + wd) * kernel_size + wh) * kernel_size + ww;
                                        
                                        neighbor_conv += input[input_idx] * weight[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                    
                    if (neighbor_conv > pooled_result) {
                        pooled_result = neighbor_conv;
                    }
                }
            }
        }
        
        output[index] = pooled_result;
    }
}

void fused_conv_transpose_pool3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int pool_kernel_size,
    int pool_stride,
    int pool_padding) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(0);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size;
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size;
    
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    fused_conv_transpose_pool3d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        pool_kernel_size,
        pool_stride,
        pool_padding
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function in the .cu file
void fused_conv_transpose_pool3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int pool_kernel_size,
    int pool_stride,
    int pool_padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_pool3d", &fused_conv_transpose_pool3d_forward, "Fused ConvTranspose3d + MaxPool3d forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_pool3d_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    max_pool1_kernel_size,
    max_pool1_stride,
    max_pool1_padding,
    max_pool1_dilation,
    max_pool1_ceil_mode,
    max_pool1_return_indices,
    max_pool2_kernel_size,
    max_pool2_stride,
    max_pool2_padding,
    max_pool2_dilation,
    max_pool2_ceil_mode,
    max_pool2_return_indices,
):
    # Fused operation: conv_transpose3d + max_pool3d
    # Note: For simplicity, we assume stride, padding are the same for all spatial dims
    conv_output_depth = (x.shape[2] - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + conv_transpose_weight.shape[2]
    conv_output_height = (x.shape[3] - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + conv_transpose_weight.shape[3]
    conv_output_width = (x.shape[4] - 1) * conv_transpose_stride[2] - 2 * conv_transpose_padding[2] + conv_transpose_weight.shape[4]
    
    # Calculate output shape after first pooling
    pooled_depth = (conv_output_depth + 2 * max_pool1_padding[0] - max_pool1_dilation[0] * (max_pool1_kernel_size[0] - 1) - 1) // max_pool1_stride[0] + 1
    pooled_height = (conv_output_height + 2 * max_pool1_padding[1] - max_pool1_dilation[1] * (max_pool1_kernel_size[1] - 1) - 1) // max_pool1_stride[1] + 1
    pooled_width = (conv_output_width + 2 * max_pool1_padding[2] - max_pool1_dilation[2] * (max_pool1_kernel_size[2] - 1) - 1) // max_pool1_stride[2] + 1
    
    pooled_output_shape = [
        x.shape[0],  # batch
        conv_transpose_weight.shape[0],  # out_channels
        pooled_depth,
        pooled_height,
        pooled_width
    ]
    
    intermediate = torch.empty(pooled_output_shape, device=x.device, dtype=x.dtype)
    
    # Call our fused CUDA kernel
    fused_ext.fused_conv_transpose_pool3d(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias,
        intermediate,
        conv_transpose_weight.shape[2],  # kernel_size (assuming cubic)
        conv_transpose_stride[0],         # stride (assuming uniform)
        conv_transpose_padding[0],        # padding (assuming uniform)
        max_pool1_kernel_size[0],         # pool_kernel_size (assuming uniform)
        max_pool1_stride[0],              # pool_stride (assuming uniform)
        max_pool1_padding[0]              # pool_padding (assuming uniform)
    )
    
    # Second max pooling operation (keep as PyTorch for now)
    x = F.max_pool3d(intermediate, 
                     kernel_size=max_pool2_kernel_size, 
                     stride=max_pool2_stride, 
                     padding=max_pool2_padding, 
                     dilation=max_pool2_dilation, 
                     ceil_mode=max_pool2_ceil_mode, 
                     return_indices=max_pool2_return_indices)
    
    # Sum reduction
    x = torch.sum(x, dim=1, keepdim=True)
    return x

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size = 5
stride = 2
padding = 2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
