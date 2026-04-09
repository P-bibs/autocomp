# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_130013/code_0.py
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

# Fused CUDA kernel that combines conv_transpose3d and post-processing
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void fused_conv_transpose3d_post_process_kernel(
    const float* input,
    const float* weight,
    const float* conv_bias,
    const float* post_bias,
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
    // Calculate global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (tid >= total_output_elements) return;
    
    // Decompose linear index to multidimensional indices
    int temp = tid;
    int out_w = temp % output_width;
    temp /= output_width;
    int out_h = temp % output_height;
    temp /= output_height;
    int out_d = temp % output_depth;
    temp /= output_depth;
    int out_ch = temp % out_channels;
    int batch_idx = temp / out_channels;
    
    // Perform convolution transpose calculation for this output position
    float sum = 0.0f;
    
    // Loop over input channels and kernel positions
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int kd = 0; kd < kernel_d; kd++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    // Calculate corresponding input position with dilation
                    int in_d = (out_d + padding_d - kd * dilation_d) / stride_d;
                    int in_h = (out_h + padding_h - kh * dilation_h) / stride_h;
                    int in_w = (out_w + padding_w - kw * dilation_w) / stride_w;
                    
                    // Check if input position is valid and aligns with stride
                    if (in_d >= 0 && in_d < input_depth &&
                        in_h >= 0 && in_h < input_height &&
                        in_w >= 0 && in_w < input_width) {
                        if ((out_d + padding_d - kd * dilation_d) % stride_d == 0 &&
                            (out_h + padding_h - kh * dilation_h) % stride_h == 0 &&
                            (out_w + padding_w - kw * dilation_w) % stride_w == 0) {
                            
                            // Calculate input and weight indices
                            int input_idx = batch_idx * (in_channels * input_depth * input_height * input_width) +
                                          in_ch * (input_depth * input_height * input_width) +
                                          in_d * (input_height * input_width) +
                                          in_h * input_width +
                                          in_w;
                                          
                            int weight_idx = out_ch * (in_channels * kernel_d * kernel_h * kernel_w) +
                                           in_ch * (kernel_d * kernel_h * kernel_w) +
                                           kd * (kernel_h * kernel_w) +
                                           kh * kernel_w +
                                           kw;
                                           
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add convolution bias
    sum += conv_bias[out_ch];
    
    // Apply post-processing: ((x + bias) + x) * x + x = (2*x + bias) * x + x
    float bias_val = post_bias[out_ch];
    float result = ((sum + bias_val) + sum) * sum + sum;
    
    // Write output
    output[tid] = result;
}

void fused_conv_transpose3d_post_process_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
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
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int out_channels = weight.size(0);
    int output_depth = output.size(2);
    int output_height = output.size(3);
    int output_width = output.size(4);
    
    // Launch configuration
    int total_output_elements = batch_size * out_channels * output_depth * output_height * output_width;
    int threads_per_block = 256;
    int blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;
    
    fused_conv_transpose3d_post_process_kernel<<<blocks, threads_per_block>>>(
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
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        dilation_d,
        dilation_h,
        dilation_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_post_process_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
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
);

torch::Tensor fused_conv_transpose3d_post_process(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
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
    int dilation_w,
    int output_depth,
    int output_height,
    int output_width
) {
    auto output = torch::empty({input.size(0), weight.size(0), output_depth, output_height, output_width}, 
                              torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    fused_conv_transpose3d_post_process_forward(input, weight, conv_bias, post_bias, output,
                                               kernel_d, kernel_h, kernel_w,
                                               stride_d, stride_h, stride_w,
                                               padding_d, padding_h, padding_w,
                                               dilation_d, dilation_h, dilation_w);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_post_process", &fused_conv_transpose3d_post_process, 
          "Fused conv transpose 3d and post-processing");
}
"""

fused_ext = load_inline(
    name='fused_conv_transpose3d_post_process_ext',
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
    # Calculate output dimensions for conv transpose 3d
    kernel_size_d = conv_transpose_weight.size(2)
    kernel_size_h = conv_transpose_weight.size(3)
    kernel_size_w = conv_transpose_weight.size(4)
    
    stride_d = conv_transpose_stride[0] if isinstance(conv_transpose_stride, (list, tuple)) else conv_transpose_stride
    stride_h = conv_transpose_stride[1] if isinstance(conv_transpose_stride, (list, tuple)) else stride_d
    stride_w = conv_transpose_stride[2] if isinstance(conv_transpose_stride, (list, tuple)) else stride_d
    
    padding_d = conv_transpose_padding[0] if isinstance(conv_transpose_padding, (list, tuple)) else conv_transpose_padding
    padding_h = conv_transpose_padding[1] if isinstance(conv_transpose_padding, (list, tuple)) else padding_d
    padding_w = conv_transpose_padding[2] if isinstance(conv_transpose_padding, (list, tuple)) else padding_d
    
    output_padding_d = conv_transpose_output_padding[0] if isinstance(conv_transpose_output_padding, (list, tuple)) else conv_transpose_output_padding
    output_padding_h = conv_transpose_output_padding[1] if isinstance(conv_transpose_output_padding, (list, tuple)) else output_padding_d
    output_padding_w = conv_transpose_output_padding[2] if isinstance(conv_transpose_output_padding, (list, tuple)) else output_padding_d
    
    dilation_d = conv_transpose_dilation[0] if isinstance(conv_transpose_dilation, (list, tuple)) else conv_transpose_dilation
    dilation_h = conv_transpose_dilation[1] if isinstance(conv_transpose_dilation, (list, tuple)) else dilation_d
    dilation_w = conv_transpose_dilation[2] if isinstance(conv_transpose_dilation, (list, tuple)) else dilation_d
    
    output_depth = (x.size(2) - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_size_d - 1) + 1 + output_padding_d
    output_height = (x.size(3) - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_size_h - 1) + 1 + output_padding_h
    output_width = (x.size(4) - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_size_w - 1) + 1 + output_padding_w
    
    # Use single fused kernel for both conv transpose and post-processing
    return fused_ext.fused_conv_transpose3d_post_process(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        bias.view(-1),
        kernel_size_d,
        kernel_size_h,
        kernel_size_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        dilation_d,
        dilation_h,
        dilation_w,
        output_depth,
        output_height,
        output_width
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
