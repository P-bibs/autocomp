# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_162105/code_0.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- CUDA Kernel Code ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_conv_transpose_bias_clamp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ add_bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const float scaling_factor,
    const float clamp_min,
    const float clamp_max
) {
    // Calculate output indices
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (out_idx >= total_elements) return;
    
    // Decode output indices
    int batch = out_idx / (out_channels * out_height * out_width);
    int remaining = out_idx % (out_channels * out_height * out_width);
    int out_c = remaining / (out_height * out_width);
    remaining = remaining % (out_height * out_width);
    int out_h = remaining / out_width;
    int out_w = remaining % out_width;
    
    // Perform transpose convolution computation for this output pixel
    float sum = 0.0f;
    
    // Calculate corresponding input region
    // For transpose conv: out_pos = stride * in_pos + kernel_pos - padding
    // So: in_pos = (out_pos + padding - kernel_pos) / stride (if integer)
    for (int k_h = 0; k_h < kernel_size; k_h++) {
        for (int k_w = 0; k_w < kernel_size; k_w++) {
            // Calculate corresponding input position
            int in_h = (out_h + padding - k_h);
            int in_w = (out_w + padding - k_w);
            
            // Check if it's a valid stride position
            if (in_h >= 0 && in_h < in_height * stride && in_h % stride == 0 &&
                in_w >= 0 && in_w < in_width * stride && in_w % stride == 0) {
                
                int orig_in_h = in_h / stride;
                int orig_in_w = in_w / stride;
                
                if (orig_in_h < in_height && orig_in_w < in_width) {
                    // Accumulate over input channels
                    for (int in_c = 0; in_c < in_channels; in_c++) {
                        int input_idx = batch * (in_channels * in_height * in_width) + 
                                       in_c * (in_height * in_width) + 
                                       orig_in_h * in_width + orig_in_w;
                                       
                        int weight_idx = out_c * (in_channels * kernel_size * kernel_size) +
                                        in_c * (kernel_size * kernel_size) +
                                        k_h * kernel_size + k_w;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add convolution bias
    sum += conv_bias[out_c];
    
    // Add additional bias
    sum += add_bias[out_c];
    
    // Apply clamping, scaling, and inverse scaling in one step
    // Since we do: clamp(x, 0, 1) * s -> clamp(result, 0, 1) / s
    // We can simplify this to direct clamping with precomputed bounds
    float final_val = sum;
    final_val = fmaxf(clamp_min, fminf(clamp_max, final_val));
    final_val = final_val * scaling_factor;
    final_val = fmaxf(clamp_min, fminf(clamp_max, final_val));
    final_val = final_val / scaling_factor;
    
    output[out_idx] = final_val;
}

void fused_conv_transpose_bias_clamp_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor conv_bias,
    const torch::Tensor add_bias,
    torch::Tensor output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const float scaling_factor,
    const float clamp_min,
    const float clamp_max
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = output.size(1);
    int out_height = output.size(2);
    int out_width = output.size(3);
    
    int total_elements = batch_size * out_channels * out_height * out_width;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    fused_conv_transpose_bias_clamp_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        add_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        scaling_factor,
        clamp_min,
        clamp_max
    );
    
    cudaDeviceSynchronize();
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_bias_clamp_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor conv_bias,
    const torch::Tensor add_bias,
    torch::Tensor output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const float scaling_factor,
    const float clamp_min,
    const float clamp_max
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_bias_clamp", &fused_conv_transpose_bias_clamp_forward, "Fused ConvTranspose2d + Bias + Clamp operations");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_bias_clamp',
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
    # Calculate output dimensions for transpose convolution
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels = conv_transpose_weight.shape[0]
    kernel_size = conv_transpose_weight.shape[2]
    
    out_height = (in_height - 1) * conv_transpose_stride + conv_transpose_dilation * (kernel_size - 1) + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride + conv_transpose_dilation * (kernel_size - 1) + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Call our fused kernel
    fused_ext.fused_conv_transpose_bias_clamp(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        bias.expand(out_channels), 
        output,
        kernel_size,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        scaling_factor,
        0.0,  # clamp_min
        1.0   # clamp_max
    )
    
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
