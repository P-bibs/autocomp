# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_113742/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'min_value', 'divisor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'min_value', 'divisor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

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
    if 'min_value' in flat_state:
        state_kwargs['min_value'] = flat_state['min_value']
    else:
        state_kwargs['min_value'] = getattr(model, 'min_value')
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
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

# CUDA kernel for fused clamp and divide operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_clamp_divide_kernel(
    const float* input,
    float* output,
    const float min_value,
    const float divisor,
    const int num_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        float val = input[idx];
        // Clamp operation
        if (val < min_value) {
            val = min_value;
        }
        // Division operation
        output[idx] = val / divisor;
    }
}

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation
) {
    const int out_depth = (in_depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    
    const int out_channels_per_group = out_channels / groups;
    const int in_channels_per_group = in_channels / groups;
    
    const int total_threads = batch_size * out_channels * out_depth * out_height * out_width;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_threads) {
        const int w = idx % out_width;
        const int h = (idx / out_width) % out_height;
        const int d = (idx / (out_width * out_height)) % out_depth;
        const int oc = (idx / (out_width * out_height * out_depth)) % out_channels;
        const int b = idx / (out_width * out_height * out_depth * out_channels);
        
        const int group = oc / out_channels_per_group;
        
        float sum = 0.0f;
        
        // Calculate input position
        for (int kd = 0; kd < kernel_size; ++kd) {
            const int in_d = d - dilation * kd + padding;
            if (in_d % stride != 0) continue;
            const int orig_d = in_d / stride;
            if (orig_d < 0 || orig_d >= in_depth) continue;
            
            for (int kh = 0; kh < kernel_size; ++kh) {
                const int in_h = h - dilation * kh + padding;
                if (in_h % stride != 0) continue;
                const int orig_h = in_h / stride;
                if (orig_h < 0 || orig_h >= in_height) continue;
                
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int in_w = w - dilation * kw + padding;
                    if (in_w % stride != 0) continue;
                    const int orig_w = in_w / stride;
                    if (orig_w < 0 || orig_w >= in_width) continue;
                    
                    for (int ic = 0; ic < in_channels_per_group; ++ic) {
                        const int input_idx = b * (in_channels * in_depth * in_height * in_width) +
                                              (group * in_channels_per_group + ic) * (in_depth * in_height * in_width) +
                                              orig_d * (in_height * in_width) +
                                              orig_h * in_width +
                                              orig_w;
                        
                        const int weight_idx = oc * (in_channels_per_group * kernel_size * kernel_size * kernel_size) +
                                               ic * (kernel_size * kernel_size * kernel_size) +
                                               kd * (kernel_size * kernel_size) +
                                               kh * kernel_size +
                                               kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        const int output_idx = b * (out_channels * out_depth * out_height * out_width) +
                               oc * (out_depth * out_height * out_width) +
                               d * (out_height * out_width) +
                               h * out_width +
                               w;
        
        output[output_idx] = sum + bias[oc];
    }
}

void fused_clamp_divide_forward(
    const torch::Tensor input,
    torch::Tensor output,
    const float min_value,
    const float divisor
) {
    const int num_elements = input.numel();
    const int threads_per_block = 256;
    const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    
    fused_clamp_divide_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        min_value,
        divisor,
        num_elements
    );
    
    cudaDeviceSynchronize();
}

void conv_transpose3d_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);
    
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);
    
    const int out_depth = (in_depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    
    const int total_threads = batch_size * out_channels * out_depth * out_height * out_width;
    const int threads_per_block = 256;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    
    conv_transpose3d_kernel<<<blocks, threads_per_block>>>(
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

# C++ bindings
cpp_source = r"""
#include <torch/extension.h>

void fused_clamp_divide_forward(
    const torch::Tensor input,
    torch::Tensor output,
    const float min_value,
    const float divisor
);

void conv_transpose3d_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_clamp_divide", &fused_clamp_divide_forward, "Fused clamp and divide operation");
    m.def("conv_transpose3d", &conv_transpose3d_forward, "3D convolution transpose operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_operations',
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
    min_value,
    divisor,
):
    # Calculate output dimensions for conv_transpose3d
    in_depth, in_height, in_width = x.shape[2], x.shape[3], x.shape[4]
    kernel_size = conv_transpose_weight.shape[2]
    
    out_depth = (in_depth - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + 1 + conv_transpose_output_padding
    out_height = (in_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + 1 + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + 1 + conv_transpose_output_padding
    
    # Create output tensor for conv_transpose3d
    conv_output = torch.empty(
        x.shape[0], 
        conv_transpose_weight.shape[1], 
        out_depth, 
        out_height, 
        out_width, 
        device=x.device, 
        dtype=x.dtype
    )
    
    # Perform conv_transpose3d using custom CUDA kernel
    fused_ext.conv_transpose3d(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        conv_output,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_groups,
        conv_transpose_dilation
    )
    
    # Create final output tensor
    final_output = torch.empty_like(conv_output)
    
    # Apply fused clamp and divide operations
    fused_ext.fused_clamp_divide(
        conv_output,
        final_output,
        min_value,
        divisor
    )
    
    return final_output

batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 24, 48, 48
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
