# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_025443/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'max_pool1_kernel_size', 'max_pool1_stride', 'max_pool1_padding', 'max_pool1_dilation', 'max_pool1_ceil_mode', 'max_pool1_return_indices', 'max_pool2_kernel_size', 'max_pool2_stride', 'max_pool2_padding', 'max_pool2_dilation', 'max_pool2_ceil_mode', 'max_pool2_return_indices']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

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
    # State for max_pool1 (nn.MaxPool3d)
    state_kwargs['max_pool1_kernel_size'] = model.max_pool1.kernel_size
    state_kwargs['max_pool1_stride'] = model.max_pool1.stride
    state_kwargs['max_pool1_padding'] = model.max_pool1.padding
    state_kwargs['max_pool1_dilation'] = model.max_pool1.dilation
    state_kwargs['max_pool1_ceil_mode'] = model.max_pool1.ceil_mode
    state_kwargs['max_pool1_return_indices'] = model.max_pool1.return_indices
    # State for max_pool2 (nn.MaxPool3d)
    state_kwargs['max_pool2_kernel_size'] = model.max_pool2.kernel_size
    state_kwargs['max_pool2_stride'] = model.max_pool2.stride
    state_kwargs['max_pool2_padding'] = model.max_pool2.padding
    state_kwargs['max_pool2_dilation'] = model.max_pool2.dilation
    state_kwargs['max_pool2_ceil_mode'] = model.max_pool2.ceil_mode
    state_kwargs['max_pool2_return_indices'] = model.max_pool2.return_indices
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

# CUDA kernel for fused operation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__device__ float atomic_add_float(float* address, float val) {
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_uint, assumed,
                        __float_as_uint(val + __uint_as_float(assumed)));
    } while (assumed != old);
    return __uint_as_float(old);
}

__global__ void fused_op_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    int max_pool1_kernel_size,
    int max_pool1_stride,
    int max_pool1_padding,
    int max_pool2_kernel_size,
    int max_pool2_stride,
    int max_pool2_padding,
    int output_depth,
    int output_height,
    int output_width
) {
    // Calculate output dimensions after each operation
    int conv_depth = (input_depth - 1) * stride + (kernel_size - 1) * dilation + 1 - 2 * padding + output_padding;
    int conv_height = (input_height - 1) * stride + (kernel_size - 1) * dilation + 1 - 2 * padding + output_padding;
    int conv_width = (input_width - 1) * stride + (kernel_size - 1) * dilation + 1 - 2 * padding + output_padding;
    
    int pool1_depth = (conv_depth + 2 * max_pool1_padding - max_pool1_kernel_size) / max_pool1_stride + 1;
    int pool1_height = (conv_height + 2 * max_pool1_padding - max_pool1_kernel_size) / max_pool1_stride + 1;
    int pool1_width = (conv_width + 2 * max_pool1_padding - max_pool1_kernel_size) / max_pool1_stride + 1;
    
    int pool2_depth = (pool1_depth + 2 * max_pool2_padding - max_pool2_kernel_size) / max_pool2_stride + 1;
    int pool2_height = (pool1_height + 2 * max_pool2_padding - max_pool2_kernel_size) / max_pool2_stride + 1;
    int pool2_width = (pool1_width + 2 * max_pool2_padding - max_pool2_kernel_size) / max_pool2_stride + 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_channels * pool2_depth * pool2_height * pool2_width;
    
    if (tid >= total_threads) return;
    
    int n = tid / (out_channels * pool2_depth * pool2_height * pool2_width);
    int c = (tid / (pool2_depth * pool2_height * pool2_width)) % out_channels;
    int od = (tid / (pool2_height * pool2_width)) % pool2_depth;
    int oh = (tid / pool2_width) % pool2_height;
    int ow = tid % pool2_width;
    
    // Calculate the corresponding position in pool1
    int pool1_d_start = od * max_pool2_stride - max_pool2_padding;
    int pool1_h_start = oh * max_pool2_stride - max_pool2_padding;
    int pool1_w_start = ow * max_pool2_stride - max_pool2_padding;
    
    float sum_val = 0.0f;
    
    // Iterate through max_pool2 kernel
    for (int kd = 0; kd < max_pool2_kernel_size; ++kd) {
        int pool1_d = pool1_d_start + kd;
        if (pool1_d < 0 || pool1_d >= pool1_depth) continue;
        
        for (int kh = 0; kh < max_pool2_kernel_size; ++kh) {
            int pool1_h = pool1_h_start + kh;
            if (pool1_h < 0 || pool1_h >= pool1_height) continue;
            
            for (int kw = 0; kw < max_pool2_kernel_size; ++kw) {
                int pool1_w = pool1_w_start + kw;
                if (pool1_w < 0 || pool1_w >= pool1_width) continue;
                
                // Find max in max_pool1
                float max_val = -INFINITY;
                
                // Calculate corresponding position in conv output
                int conv_d_start = pool1_d * max_pool1_stride - max_pool1_padding;
                int conv_h_start = pool1_h * max_pool1_stride - max_pool1_padding;
                int conv_w_start = pool1_w * max_pool1_stride - max_pool1_padding;
                
                // Iterate through max_pool1 kernel
                for (int kd1 = 0; kd1 < max_pool1_kernel_size; ++kd1) {
                    int conv_d = conv_d_start + kd1;
                    if (conv_d < 0 || conv_d >= conv_depth) continue;
                    
                    for (int kh1 = 0; kh1 < max_pool1_kernel_size; ++kh1) {
                        int conv_h = conv_h_start + kh1;
                        if (conv_h < 0 || conv_h >= conv_height) continue;
                        
                        for (int kw1 = 0; kw1 < max_pool1_kernel_size; ++kw1) {
                            int conv_w = conv_w_start + kw1;
                            if (conv_w < 0 || conv_w >= conv_width) continue;
                            
                            // Conv transpose operation
                            float conv_val = (bias) ? bias[c] : 0.0f;
                            
                            // Calculate corresponding input position
                            int in_d_start = (conv_d + padding - (kernel_size - 1) * dilation) % stride;
                            int in_h_start = (conv_h + padding - (kernel_size - 1) * dilation) % stride;
                            int in_w_start = (conv_w + padding - (kernel_size - 1) * dilation) % stride;
                            
                            if (in_d_start == 0 && in_h_start == 0 && in_w_start == 0) {
                                int in_d_base = (conv_d + padding - (kernel_size - 1) * dilation) / stride;
                                int in_h_base = (conv_h + padding - (kernel_size - 1) * dilation) / stride;
                                int in_w_base = (conv_w + padding - (kernel_size - 1) * dilation) / stride;
                                
                                // Iterate through kernel
                                for (int kd2 = 0; kd2 < kernel_size; ++kd2) {
                                    int in_d = in_d_base + kd2;
                                    if (in_d < 0 || in_d >= input_depth) continue;
                                    
                                    for (int kh2 = 0; kh2 < kernel_size; ++kh2) {
                                        int in_h = in_h_base + kh2;
                                        if (in_h < 0 || in_h >= input_height) continue;
                                        
                                        for (int kw2 = 0; kw2 < kernel_size; ++kw2) {
                                            int in_w = in_w_base + kw2;
                                            if (in_w < 0 || in_w >= input_width) continue;
                                            
                                            // Get weight index (assuming groups=1 for simplicity)
                                            int weight_idx = c * (in_channels * kernel_size * kernel_size * kernel_size) +
                                                             (in_d - in_d_base) * (kernel_size * kernel_size) +
                                                             (in_h - in_h_base) * kernel_size +
                                                             (in_w - in_w_base);
                                            
                                            int input_idx = n * (in_channels * input_depth * input_height * input_width) +
                                                            0 * (input_depth * input_height * input_width) +  // Simplified for groups=1
                                                            in_d * (input_height * input_width) +
                                                            in_h * input_width +
                                                            in_w;
                                            
                                            conv_val += input[input_idx] * weight[weight_idx];
                                        }
                                    }
                                }
                            }
                            
                            if (conv_val > max_val) {
                                max_val = conv_val;
                            }
                        }
                    }
                }
                
                sum_val += max_val;
            }
        }
    }
    
    int output_idx = n * (1 * pool2_depth * pool2_height * pool2_width) +
                     0 * (pool2_depth * pool2_height * pool2_width) +
                     od * (pool2_height * pool2_width) +
                     oh * pool2_width +
                     ow;
    
    output[output_idx] = sum_val;
}

void fused_op_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    int max_pool1_kernel_size,
    int max_pool1_stride,
    int max_pool1_padding,
    int max_pool2_kernel_size,
    int max_pool2_stride,
    int max_pool2_padding
) {
    // Calculate output dimensions
    int conv_depth = (input_depth - 1) * stride + (kernel_size - 1) * dilation + 1 - 2 * padding + output_padding;
    int conv_height = (input_height - 1) * stride + (kernel_size - 1) * dilation + 1 - 2 * padding + output_padding;
    int conv_width = (input_width - 1) * stride + (kernel_size - 1) * dilation + 1 - 2 * padding + output_padding;
    
    int pool1_depth = (conv_depth + 2 * max_pool1_padding - max_pool1_kernel_size) / max_pool1_stride + 1;
    int pool1_height = (conv_height + 2 * max_pool1_padding - max_pool1_kernel_size) / max_pool1_stride + 1;
    int pool1_width = (conv_width + 2 * max_pool1_padding - max_pool1_kernel_size) / max_pool1_stride + 1;
    
    int pool2_depth = (pool1_depth + 2 * max_pool2_padding - max_pool2_kernel_size) / max_pool2_stride + 1;
    int pool2_height = (pool1_height + 2 * max_pool2_padding - max_pool2_kernel_size) / max_pool2_stride + 1;
    int pool2_width = (pool1_width + 2 * max_pool2_padding - max_pool2_kernel_size) / max_pool2_stride + 1;
    
    int total_threads = batch_size * out_channels * pool2_depth * pool2_height * pool2_width;
    int threads_per_block = 512;
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    fused_op_kernel<<<num_blocks, threads_per_block>>>(
        input,
        weight,
        bias,
        output,
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        max_pool1_kernel_size,
        max_pool1_stride,
        max_pool1_padding,
        max_pool2_kernel_size,
        max_pool2_stride,
        max_pool2_padding,
        pool2_depth,
        pool2_height,
        pool2_width
    );
}
"""

# C++ source for PyBind11 bindings
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    int max_pool1_kernel_size,
    int max_pool1_stride,
    int max_pool1_padding,
    int max_pool2_kernel_size,
    int max_pool2_stride,
    int max_pool2_padding
);

void fused_op_wrapper(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    int max_pool1_kernel_size,
    int max_pool1_stride,
    int max_pool1_padding,
    int max_pool2_kernel_size,
    int max_pool2_stride,
    int max_pool2_padding
) {
    fused_op_forward(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        max_pool1_kernel_size,
        max_pool1_stride,
        max_pool1_padding,
        max_pool2_kernel_size,
        max_pool2_stride,
        max_pool2_padding
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_wrapper, "Fused ConvTranspose3d + MaxPool3d + Sum operation");
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
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    max_pool1_kernel_size,
    max_pool1_stride,
    max_pool1_padding,
    max_pool1_dilation,
    max_pool1_ceil_mode,
    max_pool1_return_indices,
    max_pool2_kernel_size,
    max_pool2_stride,
    max_pool2_padding,
    max_pool2_dilation,
    max_pool2_ceil_mode,
    max_pool2_return_indices,
):
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    depth = x.shape[2]
    height = x.shape[3]
    width = x.shape[4]
    out_channels = conv_transpose_weight.shape[0]
    
    # Calculate output dimensions after each operation
    conv_depth = (depth - 1) * conv_transpose_stride + (conv_transpose_weight.shape[2] - 1) * conv_transpose_dilation + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    conv_height = (height - 1) * conv_transpose_stride + (conv_transpose_weight.shape[3] - 1) * conv_transpose_dilation + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    conv_width = (width - 1) * conv_transpose_stride + (conv_transpose_weight.shape[4] - 1) * conv_transpose_dilation + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    
    pool1_depth = (conv_depth + 2 * max_pool1_padding - max_pool1_kernel_size) // max_pool1_stride + 1
    pool1_height = (conv_height + 2 * max_pool1_padding - max_pool1_kernel_size) // max_pool1_stride + 1
    pool1_width = (conv_width + 2 * max_pool1_padding - max_pool1_kernel_size) // max_pool1_stride + 1
    
    pool2_depth = (pool1_depth + 2 * max_pool2_padding - max_pool2_kernel_size) // max_pool2_stride + 1
    pool2_height = (pool1_height + 2 * max_pool2_padding - max_pool2_kernel_size) // max_pool2_stride + 1
    pool2_width = (pool1_width + 2 * max_pool2_padding - max_pool2_kernel_size) // max_pool2_stride + 1
    
    # Create output tensor
    output = torch.zeros(batch_size, 1, pool2_depth, pool2_height, pool2_width, device=x.device, dtype=x.dtype)
    
    # Call fused operation
    fused_ext.fused_op(
        x.contiguous(),
        conv_transpose_weight.contiguous(),
        conv_transpose_bias.contiguous(),
        output,
        batch_size,
        in_channels,
        out_channels,
        depth,
        height,
        width,
        conv_transpose_weight.shape[2],  # kernel_size
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_groups,
        conv_transpose_dilation,
        max_pool1_kernel_size,
        max_pool1_stride,
        max_pool1_padding,
        max_pool2_kernel_size,
        max_pool2_stride,
        max_pool2_padding
    )
    
    return output

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size = 5
stride = 2
padding = 2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
