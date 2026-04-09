# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_130013/code_6.py
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

# Optimized fused Conv3D Transpose + Post-processing CUDA kernel
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv3d_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ post_bias,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t depth_in,
    int64_t height_in,
    int64_t width_in,
    int64_t depth_out,
    int64_t height_out,
    int64_t width_out,
    int64_t kernel_d,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_d,
    int64_t padding_h,
    int64_t padding_w,
    int64_t output_padding_d,
    int64_t output_padding_h,
    int64_t output_padding_w,
    int64_t groups
) {
    // Each thread computes one output element
    int64_t output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_output_elements = batch_size * out_channels * depth_out * height_out * width_out;
    
    // Grid-stride loop to handle all output elements
    for (int64_t idx = output_idx; idx < total_output_elements; idx += gridDim.x * blockDim.x) {
        // Decompose output index to [n, c, d, h, w]
        int64_t spatial_idx = idx % (depth_out * height_out * width_out);
        int64_t channel_idx = (idx / (depth_out * height_out * width_out)) % out_channels;
        int64_t batch_idx = idx / (out_channels * depth_out * height_out * width_out);
        
        int64_t d_out = spatial_idx / (height_out * width_out);
        int64_t spatial_hw = spatial_idx % (height_out * width_out);
        int64_t h_out = spatial_hw / width_out;
        int64_t w_out = spatial_hw % width_out;
        
        // Initialize accumulator with conv bias
        float result = conv_bias[channel_idx];
        
        int64_t group_id = channel_idx / (out_channels / groups);
        int64_t in_channels_per_group = in_channels / groups;
        int64_t kernel_elements = kernel_d * kernel_h * kernel_w;
        
        // Accumulate weighted input values
        for (int64_t ic = 0; ic < in_channels_per_group; ic++) {
            for (int64_t kd = 0; kd < kernel_d; kd++) {
                for (int64_t kh = 0; kh < kernel_h; kh++) {
                    for (int64_t kw = 0; kw < kernel_w; kw++) {
                        // Compute input coordinates for transposed convolution
                        int64_t d_in = d_out / stride_d - padding_d + kd;
                        int64_t h_in = h_out / stride_h - padding_h + kh;
                        int64_t w_in = w_out / stride_w - padding_w + kw;
                        
                        // Check bounds
                        if (d_in >= 0 && d_in < depth_in && h_in >= 0 && h_in < height_in && 
                            w_in >= 0 && w_in < width_in) {
                            
                            // Compute input index
                            int64_t input_idx = batch_idx * (in_channels * depth_in * height_in * width_in) +
                                              (group_id * in_channels_per_group + ic) * (depth_in * height_in * width_in) +
                                              d_in * (height_in * width_in) +
                                              h_in * width_in +
                                              w_in;
                            
                            // Compute weight index [out_channels, in_channels/groups, kd, kh, kw]
                            int64_t weight_idx = channel_idx * (in_channels_per_group * kernel_elements) +
                                               ic * kernel_elements +
                                               kd * (kernel_h * kernel_w) +
                                               kh * kernel_w +
                                               kw;
                            
                            result += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Apply post-processing: ((x + bias) + x) * x + x = 2*x^2 + x*bias + x
        float bias_val = post_bias[channel_idx];
        float fused_result = ((result + bias_val) + result) * result + result;
        
        output[idx] = fused_result;
    }
}

void fused_conv3d_transpose_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t padding_d, int64_t padding_h, int64_t padding_w,
    int64_t output_padding_d, int64_t output_padding_h, int64_t output_padding_w,
    int64_t groups
) {
    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t depth_in = input.size(2);
    int64_t height_in = input.size(3);
    int64_t width_in = input.size(4);
    
    int64_t out_channels = weight.size(1) * groups;
    int64_t kernel_d = weight.size(2);
    int64_t kernel_h = weight.size(3);
    int64_t kernel_w = weight.size(4);
    
    // Compute output spatial dimensions
    int64_t depth_out = (depth_in - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d;
    int64_t height_out = (height_in - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    int64_t width_out = (width_in - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;
    
    int64_t total_output_elements = batch_size * out_channels * depth_out * height_out * width_out;
    
    // Use 256 threads per block and grid-stride loop
    int threads_per_block = 256;
    int blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;
    blocks = min(blocks, 65536); // Cap at max grid dimension
    
    fused_conv3d_transpose_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        post_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        depth_in, height_in, width_in,
        depth_out, height_out, width_out,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        groups
    );
    
    cudaDeviceSynchronize();
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv3d_transpose_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t padding_d, int64_t padding_h, int64_t padding_w,
    int64_t output_padding_d, int64_t output_padding_h, int64_t output_padding_w,
    int64_t groups
);

torch::Tensor fused_conv3d_transpose(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t padding_d, int64_t padding_h, int64_t padding_w,
    int64_t output_padding_d, int64_t output_padding_h, int64_t output_padding_w,
    int64_t groups
) {
    // Compute output shape
    int64_t batch_size = input.size(0);
    int64_t out_channels = weight.size(1) * groups;
    int64_t depth_out = (input.size(2) - 1) * stride_d - 2 * padding_d + weight.size(2) + output_padding_d;
    int64_t height_out = (input.size(3) - 1) * stride_h - 2 * padding_h + weight.size(3) + output_padding_h;
    int64_t width_out = (input.size(4) - 1) * stride_w - 2 * padding_w + weight.size(4) + output_padding_w;
    
    auto output = torch::empty({batch_size, out_channels, depth_out, height_out, width_out}, 
                                input.options());
    
    fused_conv3d_transpose_forward(
        input, weight, conv_bias, post_bias, output,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        groups
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv3d_transpose", &fused_conv3d_transpose, 
          "Fused conv3d transpose with post-processing");
}
"""

fused_ext = load_inline(
    name='fused_conv3d_transpose_ext',
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
    # Single fused kernel call combining conv3d transpose + post-processing
    # Extract stride, padding, and output_padding values (handle both tuple and int)
    stride_d, stride_h, stride_w = conv_transpose_stride if isinstance(conv_transpose_stride, (list, tuple)) else (conv_transpose_stride,) * 3
    padding_d, padding_h, padding_w = conv_transpose_padding if isinstance(conv_transpose_padding, (list, tuple)) else (conv_transpose_padding,) * 3
    output_padding_d, output_padding_h, output_padding_w = conv_transpose_output_padding if isinstance(conv_transpose_output_padding, (list, tuple)) else (conv_transpose_output_padding,) * 3
    
    # Flatten bias for simplified kernel indexing
    bias_flat = bias.view(-1)
    
    # Use fused kernel for both convolution and post-processing
    return fused_ext.fused_conv3d_transpose(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        bias_flat,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        conv_transpose_groups
    )

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
