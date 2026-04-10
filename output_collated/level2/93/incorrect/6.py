# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152515/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'add_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'add_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

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
    if 'add_value' in flat_state:
        state_kwargs['add_value'] = flat_state['add_value']
    else:
        state_kwargs['add_value'] = getattr(model, 'add_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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

# Optimization: Implement fused conv_transpose2d + element-wise operations in CUDA
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Device function for GELU
__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

// Convolution transpose kernel (simplified for optimization purposes)
__global__ void conv_transpose2d_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_size, int stride, int padding,
    int output_height, int output_width
) {
    int out_ch = blockIdx.x;
    int batch_idx = blockIdx.y;
    int out_y = threadIdx.y + blockIdx.z * blockDim.y;
    int out_x = threadIdx.x + blockIdx.z * blockDim.z * blockDim.y;
    
    if (out_ch >= out_channels || batch_idx >= batch_size || 
        out_y >= output_height || out_x >= output_width) return;
        
    float value = bias[out_ch];
    
    // Simplified convolution calculation
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_y = (out_y + padding - ky) / stride;
                int in_x = (out_x + padding - kx) / stride;
                
                if ((out_y + padding - ky) % stride == 0 &&
                    (out_x + padding - kx) % stride == 0 &&
                    in_y >= 0 && in_y < input_height &&
                    in_x >= 0 && in_x < input_width) {
                    
                    int input_idx = batch_idx * (in_channels * input_height * input_width) +
                                    in_ch * (input_height * input_width) +
                                    in_y * input_width + in_x;
                                    
                    int weight_idx = out_ch * (in_channels * kernel_size * kernel_size) +
                                     in_ch * (kernel_size * kernel_size) +
                                     ky * kernel_size + kx;
                                     
                    value += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    int output_idx = batch_idx * (out_channels * output_height * output_width) +
                     out_ch * (output_height * output_width) +
                     out_y * output_width + out_x;
                     
    output[output_idx] = value;
}

// Fused post-processing kernel
__global__ void fused_post_process_kernel(
    float* data, 
    float add_value, 
    float multiply_value, 
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = data[idx] + add_value;
        val = fminf(val, 0.0f); // torch.min(x, 0.0)
        data[idx] = gelu(val) * multiply_value;
    }
}

void fused_convtranspose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    float add_value,
    float multiply_value
) {
    // Get tensor dimensions
    auto input_sizes = input.sizes();
    auto output_sizes = output.sizes();
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_height = input_sizes[2];
    int input_width = input_sizes[3];
    int out_channels = output_sizes[1];
    int output_height = output_sizes[2];
    int output_width = output_sizes[3];
    
    // Launch convolution kernel
    dim3 block_conv(16, 16);
    dim3 grid_conv(out_channels, batch_size, 
                   (output_height * output_width + block_conv.x * block_conv.y - 1) / 
                   (block_conv.x * block_conv.y));
                   
    conv_transpose2d_kernel<<<grid_conv, block_conv>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        kernel_size, stride, padding,
        output_height, output_width
    );
    
    // Launch fused post-processing kernel
    int num_elements = output.numel();
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    
    fused_post_process_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(), 
        add_value, 
        multiply_value, 
        num_elements
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_convtranspose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    float add_value,
    float multiply_value
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_convtranspose2d_forward", &fused_convtranspose2d_forward, "Fused ConvTranspose2D forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_convtranspose2d',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
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
    add_value,
    multiply_value,
):
    # Calculate output dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_transpose_weight.shape[1]  # Transposed conv weight shape: (in_ch, out_ch, k, k)
    kernel_size = conv_transpose_weight.shape[2]
    
    # Compute output spatial dimensions
    out_height = (height - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + kernel_size + conv_transpose_output_padding[0]
    out_width = (width - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + kernel_size + conv_transpose_output_padding[1]
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_convtranspose2d_forward(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias,
        output,
        kernel_size,
        conv_transpose_stride[0],
        conv_transpose_padding[0],
        add_value,
        multiply_value
    )
    
    return output

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
