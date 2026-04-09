# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_132207/code_10.py
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

# Fused Conv3D Transpose + Bias + Element-wise Operations Kernel
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv3d_transpose_postproc_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ post_bias,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t in_d, int64_t in_h, int64_t in_w,
    int64_t out_d, int64_t out_h, int64_t out_w,
    int64_t kernel_d, int64_t kernel_h, int64_t kernel_w,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t pad_d, int64_t pad_h, int64_t pad_w,
    int64_t out_pad_d, int64_t out_pad_h, int64_t out_pad_w,
    int64_t groups
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_output_elements = batch_size * out_channels * out_d * out_h * out_w;
    
    if (idx >= total_output_elements) return;
    
    // Decompose output index: [b, c, d, h, w]
    int64_t remaining = idx;
    int64_t w_idx = remaining % out_w;
    remaining /= out_w;
    int64_t h_idx = remaining % out_h;
    remaining /= out_h;
    int64_t d_idx = remaining % out_d;
    remaining /= out_d;
    int64_t out_c = remaining % out_channels;
    int64_t b = remaining / out_channels;
    
    int64_t group_id = out_c / (out_channels / groups);
    int64_t in_c_per_group = in_channels / groups;
    int64_t group_start = group_id * in_c_per_group;
    
    // Accumulate convolution output
    float conv_out = 0.0f;
    
    // Compute input indices that contribute to this output position
    int64_t in_d_start = (d_idx + pad_d - kernel_d + 1 + stride_d - 1) / stride_d;
    int64_t in_h_start = (h_idx + pad_h - kernel_h + 1 + stride_h - 1) / stride_h;
    int64_t in_w_start = (w_idx + pad_w - kernel_w + 1 + stride_w - 1) / stride_w;
    
    int64_t in_d_end = (d_idx + pad_d) / stride_d + 1;
    int64_t in_h_end = (h_idx + pad_h) / stride_h + 1;
    int64_t in_w_end = (w_idx + pad_w) / stride_w + 1;
    
    // Clamp to valid input ranges
    in_d_start = max(in_d_start, (int64_t)0);
    in_h_start = max(in_h_start, (int64_t)0);
    in_w_start = max(in_w_start, (int64_t)0);
    in_d_end = min(in_d_end, in_d);
    in_h_end = min(in_h_end, in_h);
    in_w_end = min(in_w_end, in_w);
    
    for (int64_t in_d_idx = in_d_start; in_d_idx < in_d_end; ++in_d_idx) {
        for (int64_t in_h_idx = in_h_start; in_h_idx < in_h_end; ++in_h_idx) {
            for (int64_t in_w_idx = in_w_start; in_w_idx < in_w_end; ++in_w_idx) {
                // Calculate kernel indices
                int64_t kernel_d_idx = d_idx + pad_d - in_d_idx * stride_d;
                int64_t kernel_h_idx = h_idx + pad_h - in_h_idx * stride_h;
                int64_t kernel_w_idx = w_idx + pad_w - in_w_idx * stride_w;
                
                if (kernel_d_idx >= 0 && kernel_d_idx < kernel_d &&
                    kernel_h_idx >= 0 && kernel_h_idx < kernel_h &&
                    kernel_w_idx >= 0 && kernel_w_idx < kernel_w) {
                    
                    for (int64_t in_c = group_start; in_c < group_start + in_c_per_group; ++in_c) {
                        int64_t input_idx = ((b * in_channels + in_c) * in_d + in_d_idx) * in_h * in_w + 
                                           in_h_idx * in_w + in_w_idx;
                        int64_t weight_idx = (((in_c - group_start) * out_channels + out_c) * kernel_d + 
                                             (kernel_d - 1 - kernel_d_idx)) * kernel_h * kernel_w + 
                                             (kernel_h - 1 - kernel_h_idx) * kernel_w + 
                                             (kernel_w - 1 - kernel_w_idx);
                        
                        conv_out += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add convolution bias
    conv_out += conv_bias[out_c];
    
    // Apply post-processing: result = ((x + bias) + x) * x + x = x^2 + x*bias + x
    float x = conv_out + post_bias[out_c];
    float result = x * x + x * post_bias[out_c] + x;
    
    output[idx] = result;
}

void fused_conv3d_transpose_postproc_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t pad_d, int64_t pad_h, int64_t pad_w,
    int64_t out_pad_d, int64_t out_pad_h, int64_t out_pad_w,
    int64_t groups
) {
    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t in_d = input.size(2);
    int64_t in_h = input.size(3);
    int64_t in_w = input.size(4);
    int64_t out_channels = output.size(1);
    int64_t out_d = output.size(2);
    int64_t out_h = output.size(3);
    int64_t out_w = output.size(4);
    int64_t kernel_d = weight.size(2);
    int64_t kernel_h = weight.size(3);
    int64_t kernel_w = weight.size(4);
    
    int64_t num_output_elements = batch_size * out_channels * out_d * out_h * out_w;
    int threads_per_block = 256;
    int blocks = (num_output_elements + threads_per_block - 1) / threads_per_block;
    
    fused_conv3d_transpose_postproc_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        post_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w,
        groups
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv3d_transpose_postproc_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t pad_d, int64_t pad_h, int64_t pad_w,
    int64_t out_pad_d, int64_t out_pad_h, int64_t out_pad_w,
    int64_t groups
);

torch::Tensor fused_conv3d_transpose_postproc(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t pad_d, int64_t pad_h, int64_t pad_w,
    int64_t out_pad_d, int64_t out_pad_h, int64_t out_pad_w,
    int64_t groups
) {
    // Calculate output dimensions
    int64_t batch_size = input.size(0);
    int64_t in_d = input.size(2);
    int64_t in_h = input.size(3);
    int64_t in_w = input.size(4);
    int64_t out_d = (in_d - 1) * stride_d - 2 * pad_d + weight.size(2) + out_pad_d;
    int64_t out_h = (in_h - 1) * stride_h - 2 * pad_h + weight.size(3) + out_pad_h;
    int64_t out_w = (in_w - 1) * stride_w - 2 * pad_w + weight.size(4) + out_pad_w;
    int64_t out_channels = weight.size(1) * groups;
    
    auto output = torch::empty({batch_size, out_channels, out_d, out_h, out_w}, input.options());
    
    fused_conv3d_transpose_postproc_forward(
        input, weight, conv_bias, post_bias, output,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w,
        groups
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv3d_transpose_postproc", &fused_conv3d_transpose_postproc, 
          "Fused Conv3D Transpose + Bias + Element-wise Operations");
}
"""

fused_ext = load_inline(
    name='fused_conv3d_transpose_postproc_ext',
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
    """
    Fused implementation: Conv3D Transpose + Bias + Element-wise operations in a single kernel.
    Eliminates intermediate tensor materialization and reduces kernel launch overhead.
    """
    # Ensure contiguity
    x = x.contiguous()
    conv_transpose_weight = conv_transpose_weight.contiguous()
    
    # Use fused kernel that combines all operations
    return fused_ext.fused_conv3d_transpose_postproc(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        bias,
        conv_transpose_stride[0],
        conv_transpose_stride[1],
        conv_transpose_stride[2],
        conv_transpose_padding[0],
        conv_transpose_padding[1],
        conv_transpose_padding[2],
        conv_transpose_output_padding[0],
        conv_transpose_output_padding[1],
        conv_transpose_output_padding[2],
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
