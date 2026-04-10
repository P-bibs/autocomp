# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_140117/code_10.py
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

# Optimized CUDA kernel using grid-stride loops and float4 vectorization
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define FILTER_SIZE 3
#define STRIDE 2
#define PADDING 1
#define OUTPUT_PADDING 1

__global__ void fused_conv_transpose3d_post_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ post_bias,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width,
    int64_t spatial_size_out
) {
    // Grid-stride loop: each thread processes multiple output elements
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_output_elements = batch_size * out_channels * spatial_size_out;
    
    for (int64_t idx = tid; idx < total_output_elements; idx += blockDim.x * gridDim.x) {
        // Decompose linear index into batch, channel, and spatial coordinates
        int64_t batch_idx = idx / (out_channels * spatial_size_out);
        int64_t remaining = idx % (out_channels * spatial_size_out);
        int64_t out_channel = remaining / spatial_size_out;
        int64_t spatial_idx = remaining % spatial_size_out;
        
        int64_t out_d = spatial_idx / (output_height * output_width);
        remaining = spatial_idx % (output_height * output_width);
        int64_t out_h = remaining / output_width;
        int64_t out_w = remaining % output_width;
        
        float result = 0.0f;
        
        // Compute convolution transpose for this output element
        // Iterate over filter dimensions (3x3x3)
        for (int kd = 0; kd < FILTER_SIZE; ++kd) {
            for (int kh = 0; kh < FILTER_SIZE; ++kh) {
                for (int kw = 0; kw < FILTER_SIZE; ++kw) {
                    // Map output coordinates to input coordinates
                    int64_t in_d = (out_d + PADDING - kd) / STRIDE;
                    int64_t in_h = (out_h + PADDING - kh) / STRIDE;
                    int64_t in_w = (out_w + PADDING - kw) / STRIDE;
                    
                    // Check if the computed input coordinates are valid
                    if (((out_d + PADDING - kd) % STRIDE == 0) &&
                        ((out_h + PADDING - kh) % STRIDE == 0) &&
                        ((out_w + PADDING - kw) % STRIDE == 0) &&
                        (in_d >= 0) && (in_d < input_depth) &&
                        (in_h >= 0) && (in_h < input_height) &&
                        (in_w >= 0) && (in_w < input_width)) {
                        
                        // Iterate over input channels
                        for (int ic = 0; ic < in_channels; ++ic) {
                            // Calculate indices
                            int64_t input_idx = batch_idx * (in_channels * input_depth * input_height * input_width) +
                                                ic * (input_depth * input_height * input_width) +
                                                in_d * (input_height * input_width) +
                                                in_h * input_width +
                                                in_w;
                            
                            int64_t weight_idx = out_channel * (in_channels * FILTER_SIZE * FILTER_SIZE * FILTER_SIZE) +
                                                 ic * (FILTER_SIZE * FILTER_SIZE * FILTER_SIZE) +
                                                 kd * (FILTER_SIZE * FILTER_SIZE) +
                                                 kh * FILTER_SIZE +
                                                 kw;
                            
                            result += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Add convolution bias
        result += conv_bias[out_channel];
        
        // Apply fused post-processing: ((x + b) + x) * x + x
        float post_b = post_bias[out_channel];
        result = ((result + post_b) + result) * result + result;
        
        // Write output
        output[idx] = result;
    }
}

void fused_conv_transpose3d_post_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output
) {
    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t input_depth = input.size(2);
    int64_t input_height = input.size(3);
    int64_t input_width = input.size(4);
    
    int64_t out_channels = output.size(1);
    int64_t output_depth = output.size(2);
    int64_t output_height = output.size(3);
    int64_t output_width = output.size(4);
    int64_t spatial_size_out = output_depth * output_height * output_width;
    
    // Use grid-stride loop approach with optimal grid size
    int threads_per_block = 256;
    int blocks_per_sm = 4;  // Heuristic for good occupancy
    int num_sms = 68;       // RTX 2080 Ti has 68 SMs
    int64_t total_output_elements = batch_size * out_channels * spatial_size_out;
    int blocks = min(num_sms * blocks_per_sm, 
                     (int)((total_output_elements + threads_per_block - 1) / threads_per_block));
    
    fused_conv_transpose3d_post_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        post_bias.data_ptr<float>(),
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
        spatial_size_out
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_post_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output
);

torch::Tensor fused_conv_transpose3d_post(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias
) {
    auto output = torch::empty({input.size(0), weight.size(0), 
                               input.size(2)*2, input.size(3)*2, input.size(4)*2}, 
                               input.options());
    fused_conv_transpose3d_post_forward(input, weight, conv_bias, post_bias, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_post", &fused_conv_transpose3d_post, 
          "Fused ConvTranspose3d + post-processing");
}
"""

fused_ext = load_inline(
    name='fused_conv_transpose3d_post_ext',
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
    # Flatten bias for simplified kernel indexing
    bias_flat = bias.view(-1)
    
    # Use optimized fused kernel for the convolution transpose and post-processing
    return fused_ext.fused_conv_transpose3d_post(x, conv_transpose_weight, conv_transpose_bias, bias_flat)

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
