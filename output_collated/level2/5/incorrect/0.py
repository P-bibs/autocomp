# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_111953/code_1.py
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

# Optimization Plan: Fuse the activation and bias operations into the custom 
# convolution kernel to eliminate redundant global memory round-trips.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_WIDTH 16
#define CHANNEL_TILE 32

__global__ void fused_conv_transpose2d_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding
) {
    // Output tile dimensions
    const int tile_out_h = (out_height + TILE_WIDTH - 1) / TILE_WIDTH;
    const int tile_out_w = (out_width + TILE_WIDTH - 1) / TILE_WIDTH;
    
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x % tile_out_w;
    const int by = (blockIdx.x / tile_out_w) % tile_out_h;
    const int batch_idx = blockIdx.x / (tile_out_w * tile_out_h);
    const int oc_start = (blockIdx.y * CHANNEL_TILE);
    
    if (batch_idx >= batch_size) return;

    // Shared memory for intermediate results
    __shared__ float tile[TILE_WIDTH][TILE_WIDTH];
    
    // Output coordinates
    const int out_y = by * TILE_WIDTH + ty;
    const int out_x = bx * TILE_WIDTH + tx;
    
    // Process multiple output channels per block
    for (int oc = oc_start; oc < min(oc_start + CHANNEL_TILE, out_channels); ++oc) {
        float sum = 0.0f;
        
        // Perform convolution for this output pixel
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    // Calculate input position
                    int in_y = out_y - ky + padding;
                    int in_x = out_x - kx + padding;
                    
                    // Check bounds and accumulate
                    if ((in_y % stride == 0) && (in_x % stride == 0)) {
                        in_y /= stride;
                        in_x /= stride;
                        
                        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                            float inp_val = input[((batch_idx * in_channels + ic) * in_height + in_y) * in_width + in_x];
                            float wgt_val = weight[((oc * in_channels + ic) * kernel_size + ky) * kernel_size + kx];
                            sum += inp_val * wgt_val;
                        }
                    }
                }
            }
        }
        
        // Apply bias subtraction and tanh activation
        if (out_y < out_height && out_x < out_width) {
            float biased = sum - bias[oc];
            output[((batch_idx * out_channels + oc) * out_height + out_y) * out_width + out_x] = tanhf(biased);
        }
    }
}

void fused_conv_transpose2d_tanh_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int out_height = output.size(2);
    const int out_width = output.size(3);
    
    // Grid dimensions
    const int tile_out_h = (out_height + TILE_WIDTH - 1) / TILE_WIDTH;
    const int tile_out_w = (out_width + TILE_WIDTH - 1) / TILE_WIDTH;
    const int blocks_per_sample = tile_out_h * tile_out_w;
    const int channel_blocks = (out_channels + CHANNEL_TILE - 1) / CHANNEL_TILE;
    
    dim3 grid(batch_size * blocks_per_sample, channel_blocks);
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    
    fused_conv_transpose2d_tanh_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
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
        padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose2d_tanh_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv_transpose2d_tanh_forward, "Fused ConvTranspose2d with Tanh");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
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
):
    # Validate parameters
    assert conv_transpose_groups == 1, "Only conv_transpose_groups=1 is supported"
    assert conv_transpose_dilation == 1, "Only conv_transpose_dilation=1 is supported"
    assert isinstance(conv_transpose_stride, int), "Only uniform stride is supported"
    assert isinstance(conv_transpose_padding, int), "Only uniform padding is supported"
    
    # Calculate output dimensions
    N, C, H, W = x.shape
    out_channels = conv_transpose_weight.shape[0]
    kernel_size = conv_transpose_weight.shape[2]
    
    out_h = (H - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_w = (W - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    
    # Allocate output tensor
    output = torch.empty((N, out_channels, out_h, out_w), device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_op(
        x,
        conv_transpose_weight,
        bias.view(-1),
        output,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding
    )
    
    return output

batch_size = 32
in_channels  = 64  
out_channels = 64  
height = width = 256 
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
