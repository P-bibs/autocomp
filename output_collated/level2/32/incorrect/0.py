# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_010153/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'scale_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'scale_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, scales the output, and then applies a minimum operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

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
    # State for conv (nn.Conv2d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
    if 'scale_factor' in flat_state:
        state_kwargs['scale_factor'] = flat_state['scale_factor']
    else:
        state_kwargs['scale_factor'] = getattr(model, 'scale_factor')
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_conv_scale_min_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    float scale_factor
) {
    int batch_idx = blockIdx.x;
    int out_ch_idx = blockIdx.y;
    int out_h = threadIdx.x + blockIdx.z * blockDim.x;
    int out_w = threadIdx.y + blockIdx.z * blockDim.y;
    
    // Calculate output dimensions
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    if (batch_idx >= batch_size || out_ch_idx >= out_channels || out_h >= out_height || out_w >= out_width) {
        return;
    }
    
    float conv_result = 0.0f;
    
    // Perform convolution
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int in_h = out_h * stride - padding + kh * dilation;
                int in_w = out_w * stride - padding + kw * dilation;
                
                float input_val = 0.0f;
                if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                    input_val = input[((batch_idx * in_channels + in_ch) * height + in_h) * width + in_w];
                }
                
                float weight_val = weight[((out_ch_idx * in_channels + in_ch) * kernel_size + kh) * kernel_size + kw];
                conv_result += input_val * weight_val;
            }
        }
    }
    
    // Add bias
    conv_result += bias[out_ch_idx];
    
    // Scale
    conv_result *= scale_factor;
    
    // Store in temporary output
    int temp_idx = ((batch_idx * out_channels + out_ch_idx) * out_height + out_h) * out_width + out_w;
    output[temp_idx] = conv_result;
}

__global__ void min_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int out_channels,
    int out_height,
    int out_width
) {
    int batch_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int w_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || h_idx >= out_height || w_idx >= out_width) {
        return;
    }
    
    float min_val = INFINITY;
    // Calculate base index for this spatial location across all channels
    int base_idx = (batch_idx * out_channels * out_height * out_width) + (h_idx * out_width) + w_idx;
    
    for (int c = 0; c < out_channels; ++c) {
        float val = input[base_idx + c * out_height * out_width];
        if (val < min_val) {
            min_val = val;
        }
    }
    
    output[(batch_idx * out_height + h_idx) * out_width + w_idx] = min_val;
}

void fused_conv_scale_min_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    torch::Tensor& temp_output,
    int stride,
    int padding,
    int dilation,
    float scale_factor
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Launch parameters for convolution + scale
    dim3 blocks_conv(batch_size, out_channels, (out_height * out_width + 255) / 256);
    dim3 threads_conv(16, 16);
    
    fused_conv_scale_min_kernel<<<blocks_conv, threads_conv>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        temp_output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        dilation,
        scale_factor
    );
    
    // Launch parameters for min reduction
    dim3 blocks_min(batch_size, out_height);
    dim3 threads_min(min(out_width, 1024));
    
    min_reduction_kernel<<<blocks_min, threads_min>>>(
        temp_output.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        out_height,
        out_width
    );
    
    cudaDeviceSynchronize();
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_scale_min_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    torch::Tensor& temp_output,
    int stride,
    int padding,
    int dilation,
    float scale_factor
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_scale_min", &fused_conv_scale_min_forward, "Fused Conv Scale Min Forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    scale_factor,
):
    # Calculate output dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = conv_weight.shape
    
    out_height = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Create temporary output tensor for convolution result
    temp_output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Create final output tensor
    output = torch.empty(batch_size, 1, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call fused operation
    fused_ext.fused_conv_scale_min(
        x, conv_weight, conv_bias, output, temp_output,
        conv_stride, conv_padding, conv_dilation, scale_factor
    )
    
    return output

batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
