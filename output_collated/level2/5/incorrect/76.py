# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123254/code_14.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'bias_shape', 'stride', 'padding', 'output_padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

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

# ----------------------------------------------------------------------
# Custom fused CUDA kernel: convolution transpose + (x - bias) → tanh
# ----------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_WIDTH 16

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    extern __shared__ float shared_bias[];

    int tid = threadIdx.x;
    if (tid < out_channels) {
        shared_bias[tid] = bias[tid];
    }
    __syncthreads();

    int batch_idx = blockIdx.z;
    int out_ch = blockIdx.y;
    int out_row = blockIdx.x * blockDim.x + threadIdx.x;
    int out_col = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx >= batch_size || out_ch >= out_channels || out_row >= output_height || out_col >= output_width)
        return;

    float sum = 0.0f;

    // Convolution transpose implementation
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                // Calculate corresponding position in input
                int in_y = (out_row + padding - kx * stride) / stride;
                int in_x = (out_col + padding - ky * stride) / stride;

                // Check bounds and stride condition
                if ((out_row + padding - kx * stride) % stride == 0 &&
                    (out_col + padding - ky * stride) % stride == 0 &&
                    in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                    int input_idx = batch_idx * (in_channels * input_height * input_width) +
                                    in_ch * (input_height * input_width) +
                                    in_y * input_width + in_x;
                    int weight_idx = out_ch * (in_channels * kernel_size * kernel_size) +
                                     in_ch * (kernel_size * kernel_size) +
                                     (kernel_size - 1 - kx) * kernel_size + (kernel_size - 1 - ky);
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    int output_idx = batch_idx * (out_channels * output_height * output_width) +
                     out_ch * (output_height * output_width) +
                     out_row * output_width + out_col;

    output[output_idx] = tanhf(sum - shared_bias[out_ch]);
}

void fused_op_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding,
    int kernel_size
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);
    
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    dim3 grid_dim(
        (output_height + TILE_WIDTH - 1) / TILE_WIDTH,
        (output_width + TILE_WIDTH - 1) / TILE_WIDTH,
        batch_size
    );
    dim3 block_dim(TILE_WIDTH, TILE_WIDTH, 1);
    int shared_mem = out_channels * sizeof(float);

    conv_transpose2d_kernel<<<grid_dim, block_dim, shared_mem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        output_padding
    );
}
"""

# ----------------------------------------------------------------------
# C++ binding using PYBIND11
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding,
    int kernel_size
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward,
          "Fused ConvTranspose2d + bias subtraction + tanh (CUDA kernel)");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional model used by the benchmark
# ----------------------------------------------------------------------
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
):
    # Ensure all data is on the GPU
    x = x.cuda()
    conv_transpose_weight = conv_transpose_weight.cuda()
    if conv_transpose_bias is not None:
        conv_transpose_bias = conv_transpose_bias.cuda()
    bias = bias.cuda()

    # Compute output dimensions
    batch_size = x.size(0)
    in_channels = x.size(1)
    input_height = x.size(2)
    input_width = x.size(3)
    out_channels = conv_transpose_weight.size(0)
    kernel_size = conv_transpose_weight.size(2)
    
    # Handle dilation (assuming no dilation for simplicity in this kernel)
    if conv_transpose_dilation != (1, 1):
        raise ValueError("Dilation not supported in custom kernel")
    
    # Handle groups (assuming no groups for simplicity in this kernel)
    if conv_transpose_groups != 1:
        raise ValueError("Groups not supported in custom kernel")
    
    # Compute output dimensions
    stride = conv_transpose_stride[0] if isinstance(conv_transpose_stride, (tuple, list)) else conv_transpose_stride
    padding = conv_transpose_padding[0] if isinstance(conv_transpose_padding, (tuple, list)) else conv_transpose_padding
    output_padding = conv_transpose_output_padding[0] if isinstance(conv_transpose_output_padding, (tuple, list)) else conv_transpose_output_padding
    
    output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding
    output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding

    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device='cuda', dtype=x.dtype)

    # Run the fused CUDA kernel
    fused_ext.fused_op_forward(
        x,
        conv_transpose_weight,
        bias.view(-1),
        output,
        stride,
        padding,
        output_padding,
        kernel_size
    )

    return output

# ----------------------------------------------------------------------
# Test-harness boilerplate
# ----------------------------------------------------------------------
batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]
