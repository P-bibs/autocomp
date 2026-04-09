# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_012207/code_0.py
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

# Define the fused CUDA kernel
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_conv_scale_min_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float scale_factor,
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
    int out_h,
    int out_w) {
    
    int batch_idx = blockIdx.x;
    int out_y = blockIdx.y;
    int out_x = blockIdx.z;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || out_y >= out_h || out_x >= out_w)
        return;
    
    // Shared memory for min reduction
    extern __shared__ float shared_mem[];
    float* min_vals = shared_mem;
    
    float min_val = INFINITY;
    
    // Process all output channels
    for (int out_ch = tid; out_ch < out_channels; out_ch += blockDim.x) {
        float conv_result = 0.0f;
        
        // Perform convolution for this output channel
        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int in_x = out_x * stride - padding + kx * dilation;
                    int in_y = out_y * stride - padding + ky * dilation;
                    
                    if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                        int input_idx = ((batch_idx * in_channels + in_ch) * height + in_y) * width + in_x;
                        int weight_idx = ((out_ch * in_channels + in_ch) * kernel_size + ky) * kernel_size + kx;
                        conv_result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias and apply scale factor
        float result = (conv_result + bias[out_ch]) * scale_factor;
        
        // Update minimum value
        if (result < min_val) {
            min_val = result;
        }
    }
    
    // Store partial min in shared memory
    min_vals[tid] = min_val;
    __syncthreads();
    
    // Reduction in shared memory to find minimum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (min_vals[tid + s] < min_vals[tid]) {
                min_vals[tid] = min_vals[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        int output_idx = (batch_idx * out_h + out_y) * out_w + out_x;
        output[output_idx] = min_vals[0];
    }
}

void fused_conv_scale_min_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation) {
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Launch fused kernel
    dim3 grid(batch_size, out_h, out_w);
    dim3 block(256);
    int shared_mem_size = 256 * sizeof(float);
    
    fused_conv_scale_min_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale_factor,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        dilation,
        out_h,
        out_w
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_scale_min_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_scale_min", &fused_conv_scale_min_forward, "Fused conv, scale, and min reduction operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_scale_min_ext',
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
    # Validate group convolution support
    if conv_groups != 1:
        raise ValueError("Only conv_groups=1 is supported")
    
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    
    out_channels = conv_weight.shape[0]
    kernel_size = conv_weight.shape[2]
    
    # Calculate output dimensions
    out_h = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_w = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Create output tensor with correct shape (min over channels => keepdim=True with dim=1 => 1 channel)
    output = torch.empty(batch_size, 1, out_h, out_w, device=x.device, dtype=x.dtype)
    
    # Call fused operation
    fused_ext.fused_conv_scale_min(
        x, conv_weight, conv_bias, scale_factor, output, conv_stride, conv_padding, conv_dilation
    )
    
    return output

# Test parameters
batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
