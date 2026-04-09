# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_130013/code_10.py
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

# Custom 3D transposed convolution kernel + fused post-processing
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <climits>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

// CUDA kernel for 3D transposed convolution
__global__ void transposed_conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t in_depth,
    int64_t in_height,
    int64_t in_width,
    int64_t out_depth,
    int64_t out_height,
    int64_t out_width,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    // Calculate global thread index
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * out_channels * out_depth * out_height * out_width;

    if (tid >= total_elements) return;

    // Calculate indices
    int64_t tmp = tid;
    int64_t w_idx = tmp % out_width;
    tmp /= out_width;
    int64_t h_idx = tmp % out_height;
    tmp /= out_height;
    int64_t d_idx = tmp % out_depth;
    tmp /= out_depth;
    int64_t c_out = tmp % out_channels;
    int64_t batch_idx = tmp / out_channels;

    // Output coordinates
    int64_t out_d = d_idx;
    int64_t out_h = h_idx;
    int64_t out_w = w_idx;

    float value = 0.0f;

    // Iterate over input channels and kernel
    for (int64_t c_in = 0; c_in < in_channels; ++c_in) {
        for (int64_t kd = 0; kd < kernel_size; ++kd) {
            for (int64_t kh = 0; kh < kernel_size; ++kh) {
                for (int64_t kw = 0; kw < kernel_size; ++kw) {
                    // Calculate corresponding input position
                    int64_t in_d = out_d * stride - padding + kd;
                    int64_t in_h = out_h * stride - padding + kh;
                    int64_t in_w = out_w * stride - padding + kw;

                    // Check boundaries
                    if (in_d >= 0 && in_d < in_depth &&
                        in_h >= 0 && in_h < in_height &&
                        in_w >= 0 && in_w < in_width) {
                        // Input tensor index: [batch][c_in][d][h][w]
                        int64_t input_idx = batch_idx * (in_channels * in_depth * in_height * in_width) +
                                           c_in * (in_depth * in_height * in_width) +
                                           in_d * (in_height * in_width) +
                                           in_h * in_width +
                                           in_w;

                        // Weight tensor index: [c_out][c_in][kd][kh][kw]
                        int64_t weight_idx = c_out * (in_channels * kernel_size * kernel_size * kernel_size) +
                                            c_in * (kernel_size * kernel_size * kernel_size) +
                                            kd * (kernel_size * kernel_size) +
                                            kh * kernel_size +
                                            kw;

                        value += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Add bias
    value += bias[c_out];

    // Store result
    output[tid] = value;
}

// Optimized fused post-processing kernel
__global__ void fused_post_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t num_elements,
    int64_t spatial_size,
    int64_t out_channels
) {
    // Each thread handles two consecutive elements (vectorized access)
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx >= num_elements) return;

    // Compute the channel index for the bias (same for idx and idx+1 because spatial_size >> 2)
    int64_t channel_idx = (idx / spatial_size) % out_channels;
    float b = __ldg(&bias[channel_idx]);          // cached read-only load

    // ---- first element -------------------------------------------------
    float x0 = __ldg(&input[idx]);
    float r0 = ((x0 + b) + x0) * x0 + x0;         // ((x+b)+x)*x + x
    output[idx] = r0;

    // ---- second element (if it exists) --------------------------------
    if (idx + 1 < num_elements) {
        // The bias is the same for idx+1 because it stays in the same channel
        float x1 = __ldg(&input[idx + 1]);
        float r1 = ((x1 + b) + x1) * x1 + x1;
        output[idx + 1] = r1;
    }
}

void transposed_conv3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t in_depth = input.size(2);
    int64_t in_height = input.size(3);
    int64_t in_width = input.size(4);
    int64_t out_channels = weight.size(0);
    int64_t kernel_size = weight.size(2);
    
    int64_t out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int64_t out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int64_t out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    int64_t total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    // Thread configuration
    const int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    transposed_conv3d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        out_depth,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        output_padding
    );
}

void fused_post_conv_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output
) {
    int64_t num_elements = input.numel();
    int64_t spatial_size = input.size(2) * input.size(3) * input.size(4);
    int64_t out_channels = input.size(1);

    const int threads_per_block = 256;
    const int elements_per_thread = 2;               // two elements per thread
    const int64_t block_items = threads_per_block * elements_per_thread;

    int blocks = static_cast<int>((num_elements + block_items - 1) / block_items);

    fused_post_conv_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        spatial_size,
        out_channels
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void transposed_conv3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
);

void fused_post_conv_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output
);

torch::Tensor transposed_conv3d_custom(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t in_depth = input.size(2);
    int64_t in_height = input.size(3);
    int64_t in_width = input.size(4);
    int64_t out_channels = weight.size(0);
    int64_t kernel_size = weight.size(2);
    
    int64_t out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int64_t out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int64_t out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    transposed_conv3d_forward(input, weight, bias, output, stride, padding, output_padding);
    return output;
}

torch::Tensor fused_post_conv(const torch::Tensor& input, const torch::Tensor& bias) {
    auto output = torch::empty_like(input);
    fused_post_conv_forward(input, bias, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transposed_conv3d_custom", &transposed_conv3d_custom, "Custom 3D transposed convolution");
    m.def("fused_post_conv", &fused_post_conv, "Fused post-conv arithmetic (coalesced vectorized kernel)");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op_ext',
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
    bias,
):
    # Check that groups=1 and dilation=1 as we don't support other cases in our custom implementation
    if conv_transpose_groups != 1 or conv_transpose_dilation != (1, 1, 1):
        raise ValueError("Custom implementation only supports groups=1 and dilation=(1,1,1)")
        
    # Perform the 3D transposed convolution using our custom kernel
    x = fused_ext.transposed_conv3d_custom(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias,
        conv_transpose_stride[0],  # Assuming uniform stride
        conv_transpose_padding[0],  # Assuming uniform padding
        conv_transpose_output_padding[0]  # Assuming uniform output padding
    )
    
    # Flatten bias for simplified kernel indexing
    bias_flat = bias.view(-1)
    
    # Use optimized fused kernel for the intensive post-processing element-wise ops
    return fused_ext.fused_post_conv(x, bias_flat)

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
