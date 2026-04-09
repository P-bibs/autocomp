# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_031022/code_2.py
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

# Custom CUDA kernel for fused max pooling + sum reduction
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#define THREADS_PER_BLOCK 256

__global__ void fused_maxpool3d_sum_kernel(
    const float* input,
    float* output,
    int batch_size,
    int in_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int dilation_d,
    int dilation_h,
    int dilation_w
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_depth * out_height * out_width;
    
    if (out_idx >= total_output_elements) return;
    
    int w_out = out_idx % out_width;
    int h_out = (out_idx / out_width) % out_height;
    int d_out = (out_idx / (out_width * out_height)) % out_depth;
    int b = out_idx / (out_width * out_height * out_depth);
    
    int start_d = d_out * stride_d - padding_d;
    int start_h = h_out * stride_h - padding_h;
    int start_w = w_out * stride_w - padding_w;
    
    int end_d = min(start_d + (kernel_d - 1) * dilation_d + 1, in_depth);
    int end_h = min(start_h + (kernel_h - 1) * dilation_h + 1, in_height);
    int end_w = min(start_w + (kernel_w - 1) * dilation_w + 1, in_width);
    
    start_d = max(start_d, 0);
    start_h = max(start_h, 0);
    start_w = max(start_w, 0);
    
    float sum_val = 0.0f;
    
    for (int c = 0; c < in_channels; c++) {
        float max_val = -FLT_MAX;
        
        for (int kd = start_d; kd < end_d; kd += dilation_d) {
            for (int kh = start_h; kh < end_h; kh += dilation_h) {
                for (int kw = start_w; kw < end_w; kw += dilation_w) {
                    int in_idx = ((b * in_channels + c) * in_depth + kd) * in_height * in_width + kh * in_width + kw;
                    float val = input[in_idx];
                    max_val = fmaxf(max_val, val);
                }
            }
        }
        
        sum_val += max_val;
    }
    
    output[out_idx] = sum_val;
}

void fused_maxpool3d_sum_forward(
    const at::Tensor input,
    at::Tensor output,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_depth = (in_depth + 2 * padding_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    int out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    int total_output_elements = batch_size * out_depth * out_height * out_width;
    int threads = THREADS_PER_BLOCK;
    int blocks = (total_output_elements + threads - 1) / threads;
    
    fused_maxpool3d_sum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_depth,
        in_height,
        in_width,
        out_depth,
        out_height,
        out_width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ interface bindings
cpp_source = r"""
#include <torch/extension.h>

void fused_maxpool3d_sum_forward(
    const at::Tensor input,
    at::Tensor output,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_maxpool3d_sum", &fused_maxpool3d_sum_forward, "Fused 3D max pooling and sum reduction");
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

# Custom CUDA kernel for conv transpose 3d
conv_transpose3d_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_channels,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (out_idx >= total_output_elements) return;
    
    int w_out = out_idx % out_width;
    int h_out = (out_idx / out_width) % out_height;
    int d_out = (out_idx / (out_width * out_height)) % out_depth;
    int c_out = (out_idx / (out_width * out_height * out_depth)) % out_channels;
    int b = out_idx / (out_width * out_height * out_depth * out_channels);
    
    float val = (bias != nullptr) ? bias[c_out] : 0.0f;
    
    // Calculate corresponding input position
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int d_in = d_out - kd * dilation + 2 * padding;
                int h_in = h_out - kh * dilation + 2 * padding;
                int w_in = w_out - kw * dilation + 2 * padding;
                
                if (d_in % stride == 0 && h_in % stride == 0 && w_in % stride == 0) {
                    d_in /= stride;
                    h_in /= stride;
                    w_in /= stride;
                    
                    if (d_in >= 0 && d_in < in_depth && 
                        h_in >= 0 && h_in < in_height && 
                        w_in >= 0 && w_in < in_width) {
                        
                        for (int g = 0; g < groups; g++) {
                            if (c_out >= g * (out_channels / groups) && 
                                c_out < (g + 1) * (out_channels / groups)) {
                                
                                for (int c_in_group = 0; c_in_group < in_channels / groups; c_in_group++) {
                                    int c_in = g * (in_channels / groups) + c_in_group;
                                    
                                    int input_idx = ((b * in_channels + c_in) * in_depth + d_in) * in_height * in_width + h_in * in_width + w_in;
                                    int weight_idx = ((g * (out_channels / groups) + (c_out - g * (out_channels / groups))) * (in_channels / groups) + c_in_group) * kernel_size * kernel_size * kernel_size + 
                                                    (kernel_size - 1 - kd) * kernel_size * kernel_size + 
                                                    (kernel_size - 1 - kh) * kernel_size + 
                                                    (kernel_size - 1 - kw);
                                    
                                    val += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    output[out_idx] = val;
}

void conv_transpose3d_forward(
    const at::Tensor input,
    const at::Tensor weight,
    const at::Tensor bias,
    at::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_channels = weight.size(1);
    int out_depth = (in_depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    
    int total_output_elements = batch_size * out_channels * out_depth * out_height * out_width;
    int threads = THREADS_PER_BLOCK;
    int blocks = (total_output_elements + threads - 1) / threads;
    
    conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_depth,
        in_height,
        in_width,
        out_channels,
        out_depth,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        dilation
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ interface bindings for conv transpose
conv_transpose_cpp = r"""
#include <torch/extension.h>

void conv_transpose3d_forward(
    const at::Tensor input,
    const at::Tensor weight,
    const at::Tensor bias,
    at::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d", &conv_transpose3d_forward, "3D Transposed Convolution");
}
"""

# Compile the conv transpose extension
conv_transpose_ext = load_inline(
    name='conv_transpose_op',
    cpp_sources=conv_transpose_cpp,
    cuda_sources=conv_transpose3d_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Custom CUDA kernel for max pool 3d
maxpool3d_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void maxpool3d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int dilation_d,
    int dilation_h,
    int dilation_w
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * channels * out_depth * out_height * out_width;
    
    if (out_idx >= total_output_elements) return;
    
    int w_out = out_idx % out_width;
    int h_out = (out_idx / out_width) % out_height;
    int d_out = (out_idx / (out_width * out_height)) % out_depth;
    int c = (out_idx / (out_width * out_height * out_depth)) % channels;
    int b = out_idx / (out_width * out_height * out_depth * channels);
    
    int start_d = d_out * stride_d - padding_d;
    int start_h = h_out * stride_h - padding_h;
    int start_w = w_out * stride_w - padding_w;
    
    int end_d = min(start_d + (kernel_d - 1) * dilation_d + 1, in_depth);
    int end_h = min(start_h + (kernel_h - 1) * dilation_h + 1, in_height);
    int end_w = min(start_w + (kernel_w - 1) * dilation_w + 1, in_width);
    
    start_d = max(start_d, 0);
    start_h = max(start_h, 0);
    start_w = max(start_w, 0);
    
    float max_val = -FLT_MAX;
    
    for (int kd = start_d; kd < end_d; kd += dilation_d) {
        for (int kh = start_h; kh < end_h; kh += dilation_h) {
            for (int kw = start_w; kw < end_w; kw += dilation_w) {
                int in_idx = (((b * channels + c) * in_depth + kd) * in_height + kh) * in_width + kw;
                max_val = fmaxf(max_val, input[in_idx]);
            }
        }
    }
    
    output[out_idx] = max_val;
}

void maxpool3d_forward(
    const at::Tensor input,
    at::Tensor output,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_depth = (in_depth + 2 * padding_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    int out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    int total_output_elements = batch_size * channels * out_depth * out_height * out_width;
    int threads = THREADS_PER_BLOCK;
    int blocks = (total_output_elements + threads - 1) / threads;
    
    maxpool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_depth,
        in_height,
        in_width,
        out_depth,
        out_height,
        out_width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ interface bindings for max pool
maxpool_cpp = r"""
#include <torch/extension.h>

void maxpool3d_forward(
    const at::Tensor input,
    at::Tensor output,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool3d", &maxpool3d_forward, "3D Max Pooling");
}
"""

# Compile the max pool extension
maxpool_ext = load_inline(
    name='maxpool_op',
    cpp_sources=maxpool_cpp,
    cuda_sources=maxpool3d_kernel,
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
    # First operation: Conv transpose 3D
    kernel_size = conv_transpose_weight.shape[2]  # Assuming cubic kernel
    out_depth = (x.shape[2] - 1) * conv_transpose_stride + conv_transpose_dilation * (kernel_size - 1) - 2 * conv_transpose_padding + conv_transpose_output_padding + 1
    out_height = (x.shape[3] - 1) * conv_transpose_stride + conv_transpose_dilation * (kernel_size - 1) - 2 * conv_transpose_padding + conv_transpose_output_padding + 1
    out_width = (x.shape[4] - 1) * conv_transpose_stride + conv_transpose_dilation * (kernel_size - 1) - 2 * conv_transpose_padding + conv_transpose_output_padding + 1
    
    x = torch.empty(x.shape[0], conv_transpose_weight.shape[1], out_depth, out_height, out_width, device=x.device, dtype=x.dtype)
    conv_transpose_ext.conv_transpose3d(
        x, conv_transpose_weight, conv_transpose_bias, x,
        kernel_size, conv_transpose_stride, conv_transpose_padding,
        conv_transpose_output_padding, conv_transpose_groups, conv_transpose_dilation
    )
    
    # Second operation: First max pool 3D
    out_depth = (x.shape[2] + 2 * max_pool1_padding - conv_transpose_dilation * (max_pool1_kernel_size - 1) - 1) // max_pool1_stride + 1
    out_height = (x.shape[3] + 2 * max_pool1_padding - conv_transpose_dilation * (max_pool1_kernel_size - 1) - 1) // max_pool1_stride + 1
    out_width = (x.shape[4] + 2 * max_pool1_padding - conv_transpose_dilation * (max_pool1_kernel_size - 1) - 1) // max_pool1_stride + 1
    
    x_pooled = torch.empty(x.shape[0], x.shape[1], out_depth, out_height, out_width, device=x.device, dtype=x.dtype)
    maxpool_ext.maxpool3d(
        x, x_pooled,
        max_pool1_kernel_size, max_pool1_kernel_size, max_pool1_kernel_size,
        max_pool1_stride, max_pool1_stride, max_pool1_stride,
        max_pool1_padding, max_pool1_padding, max_pool1_padding,
        max_pool1_dilation, max_pool1_dilation, max_pool1_dilation
    )
    x = x_pooled
    
    # Third operation: Fused max pool 3D + sum
    out_depth = (x.shape[2] + 2 * max_pool2_padding - conv_transpose_dilation * (max_pool2_kernel_size - 1) - 1) // max_pool2_stride + 1
    out_height = (x.shape[3] + 2 * max_pool2_padding - conv_transpose_dilation * (max_pool2_kernel_size - 1) - 1) // max_pool2_stride + 1
    out_width = (x.shape[4] + 2 * max_pool2_padding - conv_transpose_dilation * (max_pool2_kernel_size - 1) - 1) // max_pool2_stride + 1
    
    output = torch.empty(x.shape[0], 1, out_depth, out_height, out_width, device=x.device, dtype=x.dtype)
    fused_ext.fused_maxpool3d_sum(
        x, output,
        max_pool2_kernel_size, max_pool2_kernel_size, max_pool2_kernel_size,
        max_pool2_stride, max_pool2_stride, max_pool2_stride,
        max_pool2_padding, max_pool2_padding, max_pool2_padding,
        max_pool2_dilation, max_pool2_dilation, max_pool2_dilation
    )
    
    return output

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
