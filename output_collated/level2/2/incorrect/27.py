# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_164254/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape', 'scaling_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias', 'scaling_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

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
    # State for conv_transpose (nn.ConvTranspose2d)
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
    if 'scaling_factor' in flat_state:
        state_kwargs['scaling_factor'] = flat_state['scaling_factor']
    else:
        state_kwargs['scaling_factor'] = getattr(model, 'scaling_factor')
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
from torch.utils.cpp_extension import load_inline

# CUDA Code: Fused kernel for Bias + Clamp + Scale + Clamp + Scale
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_post_process_kernel(float* x, const float* bias, float scaling_factor, int num_elements, int out_channels, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int c = (idx / spatial_size) % out_channels;
    
    // x = x + bias
    float val = x[idx] + bias[c];
    
    // x = clamp(x, 0, 1)
    val = fmaxf(0.0f, fminf(val, 1.0f));
    
    // x = x * scaling_factor
    val = val * scaling_factor;
    
    // x = clamp(x, 0, 1)
    val = fmaxf(0.0f, fminf(val, 1.0f));
    
    // x = x / scaling_factor
    x[idx] = val / scaling_factor;
}

// Custom ConvTranspose2d kernel
__global__ void conv_transpose2d_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding, int output_padding
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z;

    if (out_x >= (in_width - 1) * stride - 2 * padding + kernel_size + output_padding ||
        out_y >= (in_height - 1) * stride - 2 * padding + kernel_size + output_padding)
        return;

    int spatial_out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    int spatial_out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;

    float sum = 0.0f;
    for (int n = 0; n < batch_size; ++n) {
        sum = (bias != nullptr) ? bias[out_c] : 0.0f;
        for (int in_c = 0; in_c < in_channels; ++in_c) {
            for (int k_x = 0; k_x < kernel_size; ++k_x) {
                for (int k_y = 0; k_y < kernel_size; ++k_y) {
                    int in_x = (out_x + padding - k_x);
                    int in_y = (out_y + padding - k_y);
                    
                    if (in_x % stride == 0 && in_y % stride == 0) {
                        in_x /= stride;
                        in_y /= stride;
                        
                        if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
                            int input_idx = n * (in_channels * in_height * in_width) +
                                            in_c * (in_height * in_width) +
                                            in_y * in_width + in_x;
                            int weight_idx = out_c * (in_channels * kernel_size * kernel_size) +
                                             in_c * (kernel_size * kernel_size) +
                                             k_y * kernel_size + k_x;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        int output_idx = n * (out_channels * spatial_out_height * spatial_out_width) +
                         out_c * (spatial_out_height * spatial_out_width) +
                         out_y * spatial_out_width + out_x;
        output[output_idx] = sum;
    }
}

void fused_post_process(torch::Tensor x, torch::Tensor bias, float scaling_factor) {
    int num_elements = x.numel();
    int out_channels = x.size(1);
    int spatial_size = x.size(2) * x.size(3);
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    
    fused_post_process_kernel<<<blocks, threads>>>(x.data_ptr<float>(), bias.data_ptr<float>(), scaling_factor, num_elements, out_channels, spatial_size);
}

void custom_conv_transpose2d(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output,
    int stride, int padding, int output_padding, int kernel_size
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(1);
    
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    dim3 block(16, 16);
    dim3 grid((out_width + block.x - 1) / block.x, 
              (out_height + block.y - 1) / block.y, 
              out_channels);
    
    conv_transpose2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_size, stride, padding, output_padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_post_process(torch::Tensor x, torch::Tensor bias, float scaling_factor);
void custom_conv_transpose2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                             torch::Tensor output, int stride, int padding, int output_padding, int kernel_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_process", &fused_post_process, "Fused post-processing kernel");
    m.def("custom_conv_transpose2d", &custom_conv_transpose2d, "Custom ConvTranspose2d kernel");
}
"""

fused_ext = load_inline(
    name='fused_ops',
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
    scaling_factor,
):
    # Calculate output dimensions
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_size, _ = conv_transpose_weight.shape
    
    # Ensure groups and dilation are 1 (simplified implementation)
    assert conv_transpose_groups == 1, "Only groups=1 is supported"
    assert conv_transpose_dilation == 1, "Only dilation=1 is supported"
    
    # Compute output dimensions for conv transpose
    out_height = (in_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Perform custom convolution transpose
    fused_ext.custom_conv_transpose2d(
        x, conv_transpose_weight, conv_transpose_bias,
        output,
        conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding, kernel_size
    )
    
    # Apply fused post-processing
    fused_ext.fused_post_process(output, bias.reshape(-1), scaling_factor)
    
    return output

batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 128 
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
