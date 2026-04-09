# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_052229/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies HardSwish, and then ReLU.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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

# Define the CUDA kernel for fused convolution + hardswish + relu
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__device__ float hardswish_relu(float x) {
    // Hardswish: x * clamp(x + 3, 0, 6) / 6
    // ReLU: max(0, x)
    // Combined: max(0, x * clamp(x + 3, 0, 6) / 6)
    float linear = x + 3.0f;
    linear = fmaxf(0.0f, fminf(linear, 6.0f));
    return fmaxf(0.0f, x * linear / 6.0f);
}

__global__ void fused_conv_hardswish_relu_kernel(
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
    int pad,
    int stride,
    int dilation,
    int groups
) {
    // Output dimensions
    int out_height = (height + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Thread indices
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y;
    int tid_y = threadIdx.y;
    int tid_x = threadIdx.x;
    
    // Shared memory for input and weight tiles
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + 64 * 64;  // Assuming max 64x64 tile
    
    if (batch_idx >= batch_size || out_ch >= out_channels) return;
    
    int group_idx = out_ch / (out_channels / groups);
    
    // Each thread block processes one output channel for one batch
    for (int oh = 0; oh < out_height; oh++) {
        for (int ow = 0; ow < out_width; ow++) {
            float sum = 0.0f;
            
            // Convolution computation
            for (int ic = group_idx * (in_channels/groups); ic < (group_idx + 1) * (in_channels/groups); ic++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int ih = oh * stride - pad + kh * dilation;
                        int iw = ow * stride - pad + kw * dilation;
                        
                        float val = 0.0f;
                        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                            val = input[((batch_idx * in_channels + ic) * height + ih) * width + iw];
                        }
                        
                        float w_val = weight[((out_ch * (in_channels/groups) + (ic - group_idx * (in_channels/groups))) * kernel_size + kh) * kernel_size + kw];
                        sum += val * w_val;
                    }
                }
            }
            
            // Add bias
            sum += bias[out_ch];
            
            // Apply fused activation
            sum = hardswish_relu(sum);
            
            // Write output
            output[((batch_idx * out_channels + out_ch) * out_height + oh) * out_width + ow] = sum;
        }
    }
}

void fused_conv_hardswish_relu_forward(
    const at::Tensor input,
    const at::Tensor weight,
    const at::Tensor bias,
    at::Tensor output,
    int pad,
    int stride,
    int dilation,
    int groups
) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    
    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2);
    
    const auto out_height = output.size(2);
    const auto out_width = output.size(3);
    
    // Set up CUDA stream and device
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    // Launch configuration
    const dim3 blocks(batch_size, out_channels);
    const dim3 threads(16, 16);
    
    // Shared memory size (simplified)
    const int shared_mem_size = 64 * 64 * 2 * sizeof(float);
    
    fused_conv_hardswish_relu_kernel<<<blocks, threads, shared_mem_size>>>(
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
        pad,
        stride,
        dilation,
        groups
    );
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}
"""

# C++ source for PyBind11 binding
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_hardswish_relu_forward(
    const at::Tensor input,
    const at::Tensor weight,
    const at::Tensor bias,
    at::Tensor output,
    int pad,
    int stride,
    int dilation,
    int groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_hardswish_relu_forward", &fused_conv_hardswish_relu_forward, "Fused Conv + HardSwish + ReLU forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_op',
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
):
    # Calculate output dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = conv_weight.shape
    
    # Compute output spatial dimensions
    out_height = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call the fused CUDA kernel
    fused_ext.fused_conv_hardswish_relu_forward(
        x, conv_weight, conv_bias, output,
        conv_padding, conv_stride, conv_dilation, conv_groups
    )
    
    return output

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
