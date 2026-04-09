# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_101218/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a 3D convolution, applies minimum operation along a specific dimension, 
    and then applies softmax.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

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
    # State for conv (nn.Conv3d)
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
    if 'dim' in flat_state:
        state_kwargs['dim'] = flat_state['dim']
    else:
        state_kwargs['dim'] = getattr(model, 'dim')
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

# Custom CUDA kernel for fused conv3d + min reduction
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t>
__global__ void fused_conv3d_min_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_D, int input_H, int input_W,
    int output_D, int output_H, int output_W,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    int min_dim
) {
    // Calculate output indices
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * output_H * output_W;
    
    if (out_idx >= total_output_elements) return;
    
    // Decompose output index
    int w_out = out_idx % output_W;
    int h_out = (out_idx / output_W) % output_H;
    int oc = (out_idx / (output_W * output_H)) % out_channels;
    int b = out_idx / (output_W * output_H * out_channels);
    
    // Calculate input position
    int h_in = h_out * stride - padding;
    int w_in = w_out * stride - padding;
    
    // Calculate group
    int group = oc / (out_channels / groups);
    int filter_per_group = out_channels / groups;
    int oc_in_group = oc % filter_per_group;
    
    scalar_t min_val = 1e30; // Large value as initial min
    
    // Perform convolution along the min dimension (dim=2, which is depth D)
    if (min_dim == 2) {
        for (int d_out = 0; d_out < output_D; ++d_out) {
            int d_in = d_out * stride - padding;
            scalar_t sum = 0.0;
            
            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kc = 0; kc < kernel_size; ++kc) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        for (int ic = 0; ic < in_channels / groups; ++ic) {
                            int input_c = group * (in_channels / groups) + ic;
                            
                            int d_coord = d_in + kd * dilation;
                            int h_coord = h_in + kc * dilation;
                            int w_coord = w_in + kw * dilation;
                            
                            if (d_coord >= 0 && d_coord < input_D &&
                                h_coord >= 0 && h_coord < input_H && 
                                w_coord >= 0 && w_coord < input_W) {
                                
                                int input_idx = b * (in_channels * input_D * input_H * input_W) +
                                              input_c * (input_D * input_H * input_W) +
                                              d_coord * (input_H * input_W) +
                                              h_coord * input_W + 
                                              w_coord;
                                              
                                int weight_idx = oc * (in_channels / groups * kernel_size * kernel_size * kernel_size) +
                                               ic * (kernel_size * kernel_size * kernel_size) +
                                               kd * (kernel_size * kernel_size) +
                                               kc * kernel_size +
                                               kw;
                                               
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
            
            // Add bias
            sum += bias[oc];
            
            if (sum < min_val) {
                min_val = sum;
            }
        }
    }
    
    output[out_idx] = min_val;
}

void fused_conv3d_min_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups,
    int min_dim
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_D = input.size(2);
    auto input_H = input.size(3);
    auto input_W = input.size(4);
    
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2); // Assumes cubic kernel
    
    auto output_D = (input_D + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto output_H = (input_H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto output_W = (input_W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto total_output_elements = batch_size * out_channels * output_H * output_W;
    
    const int threads = 256;
    const int blocks = (total_output_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_conv3d_min_forward", ([&] {
        fused_conv3d_min_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_D, input_H, input_W,
            output_D, output_H, output_W,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            min_dim
        );
    }));
    
    cudaDeviceSynchronize();
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void fused_conv3d_min_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups,
    int min_dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv3d_min", &fused_conv3d_min_forward, "Fused Conv3D and Min operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv3d_min',
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
    dim,
):
    # First, perform fused conv3d + min operation
    # Calculate output dimensions
    kernel_size = conv_weight.shape[2]  # assuming cubic kernel
    output_D = (x.shape[2] + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    output_H = (x.shape[3] + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    output_W = (x.shape[4] + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Create output tensor for conv + min (after reduction along dim=2)
    conv_min_output = torch.empty(x.shape[0], conv_weight.shape[0], output_H, output_W, 
                                  dtype=x.dtype, device=x.device)
    
    # Call fused kernel
    fused_ext.fused_conv3d_min(x, conv_weight, conv_bias, conv_min_output, 
                               conv_stride, conv_padding, conv_dilation, conv_groups, dim)
    
    # Apply softmax
    result = torch.softmax(conv_min_output, dim=1)
    
    return result

batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
dim = 2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]
