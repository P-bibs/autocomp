# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_134740/code_15.py
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

# ----------------------------------------------------------------------
# Optimised CUDA kernel: float4 vectorisation, direct global-memory bias,
# cheap channel indexing (shift+mask), no shared-memory copy.
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_post_conv_kernel(
    const float4* __restrict__ input,
    const float* __restrict__ bias,
    float4* __restrict__ output,
    int64_t num_elements_float4,
    int64_t spatial_size,
    int64_t out_channels,
    int shift,                // >0 if spatial_size is power-of-two, else -1
    unsigned int mask         // out_channels-1 (valid only when shift>=0)
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements_float4) {
        float4 x_vec = input[idx];
        float4 result;

        // base index of the first element of this float4 vector
        int64_t base = idx * 4;

        // ---- channel index per element (fast path for power-of-two) ----
        int c0, c1, c2, c3;
        if (shift >= 0) {                         // spatial_size == 2^shift
            c0 = ( (base     >> shift) ) & mask;
            c1 = ( (base + 1 >> shift) ) & mask;
            c2 = ( (base + 2 >> shift) ) & mask;
            c3 = ( (base + 3 >> shift) ) & mask;
        } else {                                   // generic case
            c0 = ( (base     / spatial_size) ) % out_channels;
            c1 = ( ((base + 1) / spatial_size) ) % out_channels;
            c2 = ( ((base + 2) / spatial_size) ) % out_channels;
            c3 = ( ((base + 3) / spatial_size) ) % out_channels;
        }

        // ---- read bias directly from global memory (read-only cache) ----
        float b0 = __ldg(&bias[c0]);
        float b1 = __ldg(&bias[c1]);
        float b2 = __ldg(&bias[c2]);
        float b3 = __ldg(&bias[c3]);

        // ---- fused arithmetic: ((x + bias) + x) * x + x ----
        result.x = ((x_vec.x + b0) + x_vec.x) * x_vec.x + x_vec.x;
        result.y = ((x_vec.y + b1) + x_vec.y) * x_vec.y + x_vec.y;
        result.z = ((x_vec.z + b2) + x_vec.z) * x_vec.z + x_vec.z;
        result.w = ((x_vec.w + b3) + x_vec.w) * x_vec.w + x_vec.w;

        output[idx] = result;
    }
}

void fused_post_conv_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output)
{
    const int64_t num_elems          = input.numel();          // total floats
    const int64_t num_elems_float4   = num_elems / 4;          // guaranteed divisible
    const int64_t spatial_size       = input.size(2) * input.size(3) * input.size(4);
    const int64_t out_channels       = input.size(1);

    // ---- decide whether we can use the cheap shift+mask path ----
    int shift = -1;
    unsigned int mask = 0;
    if ((spatial_size & (spatial_size - 1)) == 0) {            // power-of-two
        shift = __builtin_ctzll(spatial_size);
        mask  = static_cast<unsigned int>(out_channels - 1);
    }

    const int threads = 256;
    const int blocks  = (num_elems_float4 + threads - 1) / threads;

    const float4* in_ptr  = reinterpret_cast<const float4*>(input.data_ptr<float>());
    float4*       out_ptr = reinterpret_cast<float4*>(output.data_ptr<float>());

    fused_post_conv_kernel<<<blocks, threads>>>(
        in_ptr,
        bias.data_ptr<float>(),
        out_ptr,
        num_elems_float4,
        spatial_size,
        out_channels,
        shift,
        mask
    );
}
"""

# ----------------------------------------------------------------------
# C++ binding (PYBIND11) – no custom “function” argument
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_post_conv_forward(const torch::Tensor& input,
                             const torch::Tensor& bias,
                             torch::Tensor& output);

torch::Tensor fused_post_conv(const torch::Tensor& input,
                              const torch::Tensor& bias) {
    auto output = torch::empty_like(input);
    fused_post_conv_forward(input, bias, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv", &fused_post_conv,
          "Fused post-conv arithmetic: float4 vectorisation + direct global bias");
}
"""

fused_ext = load_inline(
    name='fused_post_conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Custom ConvTranspose3d implementation using CUDA
# ----------------------------------------------------------------------
conv_transpose3d_cuda = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    if (idx >= total_elements) return;
    
    int w = idx % output_w;
    idx /= output_w;
    int h = idx % output_h;
    idx /= output_h;
    int d = idx % output_d;
    idx /= output_d;
    int oc = idx % out_channels;
    int b = idx / out_channels;
    
    float sum = 0.0f;
    
    // Calculate input position considering stride and padding
    int start_id = (d + padding - (kernel_size - 1) * dilation) % stride;
    int start_ih = (h + padding - (kernel_size - 1) * dilation) % stride;
    int start_iw = (w + padding - (kernel_size - 1) * dilation) % stride;
    
    for (int kd = start_id; kd < kernel_size; kd += stride) {
        for (int kh = start_ih; kh < kernel_size; kh += stride) {
            for (int kw = start_iw; kw < kernel_size; kw += stride) {
                int id = (d + padding - kd * dilation) / stride;
                int ih = (h + padding - kh * dilation) / stride;
                int iw = (w + padding - kw * dilation) / stride;
                
                if (id >= 0 && id < input_d && ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                    int ic = oc; // Assuming groups == out_channels (depthwise-like)
                    if (groups == 1) ic = 0; // All input channels contribute to each output channel
                    
                    int input_idx = ((((b * in_channels + ic) * input_d + id) * input_h + ih) * input_w + iw);
                    int weight_idx = ((((oc * in_channels + ic) * kernel_size + kd) * kernel_size + kh) * kernel_size + kw);
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    if (bias) {
        sum += bias[oc];
    }
    
    output[idx] = sum;
}

torch::Tensor conv_transpose3d_cuda_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
) {
    // Input: (N, C_in, D_in, H_in, W_in)
    // Weight: (C_in, C_out/groups, kD, kH, kW)
    // Output: (N, C_out, D_out, H_out, W_out)
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_d = input.size(2);
    int input_h = input.size(3);
    int input_w = input.size(4);
    
    int out_channels = weight.size(1) * groups;
    int kernel_size = weight.size(2);
    
    // Calculate output dimensions
    int output_d = (input_d - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int output_h = (input_h - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int output_w = (input_w - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_d, output_h, output_w}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    conv_transpose3d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        dilation
    );
    
    return output;
}
"""

conv_transpose3d_cpp = r"""
#include <torch/extension.h>

torch::Tensor conv_transpose3d_cuda_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
);

torch::Tensor conv_transpose3d_custom(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups,
    int64_t dilation
) {
    return conv_transpose3d_cuda_forward(input, weight, bias, 
                                        static_cast<int>(stride),
                                        static_cast<int>(padding),
                                        static_cast<int>(output_padding),
                                        static_cast<int>(groups),
                                        static_cast<int>(dilation));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d_custom", &conv_transpose3d_custom, "Custom ConvTranspose3d implementation");
}
"""

conv_transpose_ext = load_inline(
    name='conv_transpose3d_ext',
    cpp_sources=conv_transpose3d_cpp,
    cuda_sources=conv_transpose3d_cuda,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional model used by the benchmark harness
# ----------------------------------------------------------------------
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
    # ---- custom transposed convolution ----
    x = conv_transpose_ext.conv_transpose3d_custom(
        x, conv_transpose_weight, conv_transpose_bias,
        conv_transpose_stride, conv_transpose_padding,
        conv_transpose_output_padding, conv_transpose_groups,
        conv_transpose_dilation
    )

    # ---- flatten bias and ensure correct memory layout ----
    bias_flat = bias.view(-1)
    x = x.contiguous()

    # ---- invoke the optimized fused kernel ----
    return fused_ext.fused_post_conv(x, bias_flat)


# ----------------------------------------------------------------------
# Helper functions required by the evaluation harness
# ----------------------------------------------------------------------
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
