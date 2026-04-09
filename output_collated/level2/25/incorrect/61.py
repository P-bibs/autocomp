# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_091623/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_conv_min_tanh_kernel(
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
    int groups) {
    
    // Calculate output dimensions
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Get thread indices
    int batch_idx = blockIdx.x;
    int out_y = blockIdx.y;
    int out_x = blockIdx.z;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || out_y >= out_height || out_x >= out_width) return;
    
    // Shared memory for reduction
    extern __shared__ float shared_data[];
    
    // Each thread processes one channel
    float min_val = INFINITY;
    
    // Group size calculations
    int group_size = out_channels / groups;
    int group_id = tid / group_size;
    int ch_start = group_id * group_size;
    int ch_end = min(ch_start + group_size, out_channels);
    
    // Process channels assigned to this thread
    for (int out_ch = ch_start + (tid % group_size); out_ch < ch_end; out_ch += blockDim.x) {
        // Convolution computation
        float sum = 0.0f;
        
        // Get group for this output channel
        int group = out_ch * groups / out_channels;
        int in_ch_start = group * in_channels / groups;
        int in_ch_end = (group + 1) * in_channels / groups;
        
        // Convolution loop
        for (int in_ch = in_ch_start; in_ch < in_ch_end; ++in_ch) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int in_y = out_y * stride - padding + ky * dilation;
                    int in_x = out_x * stride - padding + kx * dilation;
                    
                    if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                        int input_idx = batch_idx * in_channels * height * width + 
                                       in_ch * height * width + 
                                       in_y * width + in_x;
                        int weight_idx = out_ch * (in_channels / groups) * kernel_size * kernel_size + 
                                        (in_ch - in_ch_start) * kernel_size * kernel_size + 
                                        ky * kernel_size + kx;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias
        sum += bias[out_ch];
        
        // Apply tanh twice
        sum = tanhf(sum);
        sum = tanhf(sum);
        
        // Update minimum
        min_val = fminf(min_val, sum);
    }
    
    // Store in shared memory
    shared_data[tid] = min_val;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = fminf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    
    // Write output
    if (tid == 0) {
        int output_idx = batch_idx * out_height * out_width + 
                        out_y * out_width + out_x;
        output[output_idx] = shared_data[0];
    }
}

void fused_conv_min_tanh_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Launch parameters
    dim3 block_size(256);
    dim3 grid_size(batch_size, out_height, out_width);
    size_t shared_mem_size = block_size.x * sizeof(float);
    
    // Launch kernel
    fused_conv_min_tanh_kernel<<<grid_size, block_size, shared_mem_size>>>(
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
        groups
    );
    
    cudaDeviceSynchronize();
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_min_tanh_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int dilation,
    int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_tanh", &fused_conv_min_tanh_forward, "Fused Conv + Min + Tanh forward pass");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_min_tanh_ext',
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
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_weight.shape[0]
    kernel_size = conv_weight.shape[2]
    
    # Calculate output dimensions
    out_height = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, 1, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_conv_min_tanh(
        x, conv_weight, conv_bias, output,
        conv_stride, conv_padding, conv_dilation, conv_groups
    )
    
    return output

batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
