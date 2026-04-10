# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160018/code_4.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# The CUDA source includes a tiled conv_transpose2d fused with add, min, gelu, and multiply operations.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 16
#define KERNEL_SIZE 4

__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_transpose2d_kernel(
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
    float add_val,
    float mul_val
) {
    // Shared memory for weights
    extern __shared__ float shared_weights[];

    int out_height = (height - 1) * stride + kernel_size;
    int out_width = (width - 1) * stride + kernel_size;

    int batch_idx = blockIdx.z;
    int out_ch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || out_ch_idx >= out_channels || out_x >= out_height * out_width) return;

    int out_y_base = out_x / out_width;
    int out_x_base = out_x % out_width;

    // Load weights into shared memory
    int weight_threads = kernel_size * kernel_size * in_channels;
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < weight_threads; i += blockDim.y * blockDim.x) {
        shared_weights[i] = weight[out_ch_idx * weight_threads + i];
    }
    __syncthreads();

    float sum = 0.0f;

    if (out_y_base >= 0 && out_y_base < out_height && out_x_base >= 0 && out_x_base < out_width) {
        // Compute input coordinates
        int in_y = out_y_base / stride;
        int in_x = out_x_base / stride;

        if (in_y < height && in_x < width && in_y >= 0 && in_x >= 0) {
            int ky_start = out_y_base % stride;
            int kx_start = out_x_base % stride;

            for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
                float input_val = input[batch_idx * (in_channels * height * width) + 
                                        in_ch * (height * width) + 
                                        in_y * width + in_x];

                for (int ky = ky_start; ky < kernel_size; ky += stride) {
                    for (int kx = kx_start; kx < kernel_size; kx += stride) {
                        int weight_idx = in_ch * kernel_size * kernel_size + ky * kernel_size + kx;
                        sum += input_val * shared_weights[weight_idx];
                    }
                }
            }
        }

        // Add bias
        sum += bias[out_ch_idx];

        // Apply fused operations: add -> min -> gelu -> multiply
        float val = sum + add_val;
        val = fminf(val, 0.0f);
        val = fast_gelu(val);
        output[batch_idx * (out_channels * out_height * out_width) + 
               out_ch_idx * (out_height * out_width) + 
               out_y_base * out_width + out_x_base] = val * mul_val;
    }
}

void fused_conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    float add_val,
    float mul_val
) {
    int out_height = (height - 1) * stride + kernel_size;
    int out_width = (width - 1) * stride + kernel_size;

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_height * out_width + block.x - 1) / block.x, 
              (out_channels + block.y - 1) / block.y, 
              batch_size);

    int shared_mem_size = kernel_size * kernel_size * in_channels * sizeof(float);
    
    fused_conv_transpose2d_kernel<<<grid, block, shared_mem_size>>>(
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
        add_val,
        mul_val
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    float add_val,
    float mul_val
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose2d", &fused_conv_transpose2d_forward, "Fused conv_transpose2d with add-min-gelu-mul");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose2d_op',
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
    add_value,
    multiply_value,
):
    # Get dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_transpose_weight.shape[0]
    kernel_size = conv_transpose_weight.shape[2]
    
    # Calculate output dimensions
    out_height = (height - 1) * conv_transpose_stride + kernel_size
    out_width = (width - 1) * conv_transpose_stride + kernel_size
    
    # Create output tensor
    out = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_conv_transpose2d(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        out,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        conv_transpose_stride,
        float(add_value),
        float(multiply_value)
    )
    
    return out

# Constants provided by original context
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
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
