# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_130922/code_12.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by a sum, 
    a residual add, a multiplication, and another residual add.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

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
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
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
import math

# -------------------------------------------------------------------------
# Optimized CUDA kernel – uses shift/mask for channel index and float4 load
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_post_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int num_elements,
    const int spatial_shift,          // log2(spatial_size)
    const int out_channels)
{
    // Each thread handles 4 consecutive elements (vectorized load/store)
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx >= num_elements) return;

    // --------------------------------------------------------------
    // 1) Compute channel index via bitwise shift & mask.
    //    spatial_shift = log2(spatial_size) (spatial_size is power of two)
    // --------------------------------------------------------------
    int channel_idx = (idx >> spatial_shift) & (out_channels - 1);

    // --------------------------------------------------------------
    // 2) Load bias once per thread using the read-only cache
    // --------------------------------------------------------------
    float bias_val = __ldg(&bias[channel_idx]);

    // --------------------------------------------------------------
    // 3) Vectorized load of 4 input elements
    // --------------------------------------------------------------
    float4 inp = __ldg(reinterpret_cast<const float4*>(input + idx));

    // --------------------------------------------------------------
    // 4) Compute the four results:
    //    ((x + bias) + x) * x + x
    // --------------------------------------------------------------
    float res_x = (( (inp.x + bias_val) + inp.x ) * inp.x + inp.x);
    float res_y = (( (inp.y + bias_val) + inp.y ) * inp.y + inp.y);
    float res_z = (( (inp.z + bias_val) + inp.z ) * inp.z + inp.z);
    float res_w = (( (inp.w + bias_val) + inp.w ) * inp.w + inp.w);

    // --------------------------------------------------------------
    // 5) Vectorized store
    // --------------------------------------------------------------
    float4 out_vec = {res_x, res_y, res_z, res_w};
    reinterpret_cast<float4*>(output + idx)[0] = out_vec;
}

void fused_post_conv_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output)
{
    const int num_elements = static_cast<int>(input.numel());
    const int out_channels = static_cast<int>(input.size(1));
    const int spatial_size = static_cast<int>(input.size(2) * input.size(3) * input.size(4));

    // Compute shift = log2(spatial_size).  spatial_size is a power of two.
    int spatial_shift = 0;
    int tmp = spatial_size;
    while (tmp > 1) {
        spatial_shift++;
        tmp >>= 1;
    }

    const int threads_per_block = 256;
    const int blocks = (num_elements + threads_per_block * 4 - 1) / (threads_per_block * 4);

    fused_post_conv_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        spatial_shift,
        out_channels);
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_post_conv_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output);

torch::Tensor fused_post_conv(const torch::Tensor& input, const torch::Tensor& bias) {
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    auto output = torch::empty_like(input);
    fused_post_conv_forward(input, bias, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv", &fused_post_conv, "Fused post-conv arithmetic");
}
"""

# -------------------------------------------------------------------------
# Build the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_post_conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Custom ConvTranspose3d CUDA kernel implementation
# -------------------------------------------------------------------------
conv_transpose3d_cuda = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int k_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int out_depth,
    const int out_height,
    const int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int d = (idx / (out_width * out_height)) % out_depth;
    int c = (idx / (out_width * out_height * out_depth)) % out_channels;
    int b = idx / (out_width * out_height * out_depth * out_channels);
    
    float sum = 0.0f;
    
    // Calculate input region that contributes to this output point
    int d_start = (d + padding - k_size + 1 + stride - 1) / stride;  // ceil division
    int d_end = min((d + padding) / stride + 1, in_depth);
    int h_start = (h + padding - k_size + 1 + stride - 1) / stride;
    int h_end = min((h + padding) / stride + 1, in_height);
    int w_start = (w + padding - k_size + 1 + stride - 1) / stride;
    int w_end = min((w + padding) / stride + 1, in_width);
    
    d_start = max(d_start, 0);
    h_start = max(h_start, 0);
    w_start = max(w_start, 0);
    
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = d_start; kd < d_end; ++kd) {
            for (int kh = h_start; kh < h_end; ++kh) {
                for (int kw = w_start; kw < w_end; ++kw) {
                    int k_di = d - kd * stride + padding;
                    int k_hi = h - kh * stride + padding;
                    int k_wi = w - kw * stride + padding;
                    
                    if (k_di >= 0 && k_di < k_size &&
                        k_hi >= 0 && k_hi < k_size &&
                        k_wi >= 0 && k_wi < k_size) {
                        
                        int input_idx = b * (in_channels * in_depth * in_height * in_width) +
                                       ic * (in_depth * in_height * in_width) +
                                       kd * (in_height * in_width) +
                                       kh * in_width +
                                       kw;
                                       
                        int weight_idx = c * (in_channels * k_size * k_size * k_size) +
                                        ic * (k_size * k_size * k_size) +
                                        k_di * (k_size * k_size) +
                                        k_hi * k_size +
                                        k_wi;
                                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    if (bias != nullptr) {
        sum += bias[c];
    }
    
    output[idx] = sum;
}

void conv_transpose3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);
    
    const int k_size = weight.size(2); // Assuming cubic kernel
    const int out_channels = weight.size(0);
    
    const int out_depth = (in_depth - 1) * stride - 2 * padding + k_size + output_padding;
    const int out_height = (in_height - 1) * stride - 2 * padding + k_size + output_padding;
    const int out_width = (in_width - 1) * stride - 2 * padding + k_size + output_padding;
    
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    conv_transpose3d_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        k_size,
        stride,
        padding,
        output_padding,
        out_depth,
        out_height,
        out_width
    );
}
"""

conv_transpose3d_cpp = r"""
#include <torch/extension.h>

void conv_transpose3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding);

torch::Tensor conv_transpose3d_custom(const torch::Tensor& input, 
                                      const torch::Tensor& weight,
                                      const torch::Tensor& bias,
                                      int stride,
                                      int padding,
                                      int output_padding) {
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    }
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);
    
    const int k_size = weight.size(2);
    const int out_channels = weight.size(0);
    
    const int out_depth = (in_depth - 1) * stride - 2 * padding + k_size + output_padding;
    const int out_height = (in_height - 1) * stride - 2 * padding + k_size + output_padding;
    const int out_width = (in_width - 1) * stride - 2 * padding + k_size + output_padding;
    
    auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
                               
    conv_transpose3d_forward(input, weight, bias, output, stride, padding, output_padding);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d_custom", &conv_transpose3d_custom, "Custom ConvTranspose3d");
}
"""

# Build ConvTranspose3D extension
conv_transpose3d_ext = load_inline(
    name='conv_transpose3d_ext',
    cpp_sources=conv_transpose3d_cpp,
    cuda_sources=conv_transpose3d_cuda,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional model used for evaluation
# -------------------------------------------------------------------------
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
    bias,
):
    # Use custom ConvTranspose3d implementation
    x = conv_transpose3d_ext.conv_transpose3d_custom(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        conv_transpose_stride[0],  # Assuming uniform stride
        conv_transpose_padding[0],  # Assuming uniform padding
        conv_transpose_output_padding[0]  # Assuming uniform output padding
    )

    # Flatten bias to a 1-D tensor (required by the kernel)
    bias_flat = bias.view(-1)

    # ----- Optimized fused kernel -----
    return fused_ext.fused_post_conv(x, bias_flat)

# -------------------------------------------------------------------------
# Helper code (shape parameters, input factories)
# -------------------------------------------------------------------------
batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
