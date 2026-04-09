# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_011649/code_2.py
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

# Custom CUDA kernel for fused Conv2D + Scaling + Min Reduction
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

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
    float scale_factor,
    int out_height,
    int out_width
) {
    // Calculate indices
    int batch_idx = blockIdx.x;
    int out_h = blockIdx.y;
    int out_w = blockIdx.z;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || out_h >= out_height || out_w >= out_width) return;
    
    // Shared memory to store intermediate min values
    extern __shared__ float sdata[];
    
    // Calculate input offsets
    int input_batch_offset = batch_idx * in_channels * height * width;
    int weight_offset = 0;
    
    float min_val = INFINITY;
    
    // Each thread processes some output channels
    for (int out_c = tid; out_c < out_channels; out_c += blockDim.x) {
        float sum = 0.0f;
        
        // Perform convolution for this output channel
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                for (int in_c = 0; in_c < in_channels; in_c++) {
                    // Calculate input position
                    int in_y = out_h * stride - padding + ky * dilation;
                    int in_x = out_w * stride - padding + kx * dilation;
                    
                    float val = 0.0f;
                    if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                        int input_idx = input_batch_offset + in_c * height * width + in_y * width + in_x;
                        val = input[input_idx];
                    }
                    
                    int weight_idx = out_c * in_channels * kernel_size * kernel_size + 
                                    in_c * kernel_size * kernel_size + 
                                    ky * kernel_size + kx;
                    sum += val * weight[weight_idx];
                }
            }
        }
        
        // Add bias
        sum += bias[out_c];
        
        // Apply scale factor
        sum *= scale_factor;
        
        // Update minimum
        if (sum < min_val) min_val = sum;
    }
    
    // Store in shared memory
    sdata[tid] = min_val;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] < sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        int output_idx = batch_idx * out_height * out_width + out_h * out_width + out_w;
        output[output_idx] = sdata[0];
    }
}

void fused_conv_scale_min_forward(
    const at::Tensor input,
    const at::Tensor weight,
    const at::Tensor bias,
    at::Tensor output,
    int stride,
    int padding,
    int dilation,
    float scale_factor,
    int out_height,
    int out_width
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    // Define block and grid dimensions
    dim3 grid(batch_size, out_height, out_width);
    dim3 block(256); // Threads per block
    
    // Shared memory size
    int shared_mem_size = block.x * sizeof(float);
    
    fused_conv_scale_min_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
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
        scale_factor,
        out_height,
        out_width
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_scale_min_forward(
    const at::Tensor input,
    const at::Tensor weight,
    const at::Tensor bias,
    at::Tensor output,
    int stride,
    int padding,
    int dilation,
    float scale_factor,
    int out_height,
    int out_width
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_scale_min_forward", &fused_conv_scale_min_forward, "Fused Conv2D + Scale + Min Reduction Forward");
}
"""

# Compile the CUDA extension
fused_ext = load_inline(
    name='fused_conv_scale_min',
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
    # Validate inputs
    assert conv_groups == 1, "Only conv_groups=1 is supported"
    
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = conv_weight.shape
    
    # Calculate output dimensions
    out_height = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Create output tensor
    output = torch.empty((batch_size, 1, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_conv_scale_min_forward(
        x, conv_weight, conv_bias, output,
        conv_stride, conv_padding, conv_dilation,
        scale_factor, out_height, out_width
    )
    
    return output

# Constants for testing
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
