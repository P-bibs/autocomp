# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160431/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv1d_weight', 'conv1d_bias', 'conv1d_stride', 'conv1d_padding', 'conv1d_dilation', 'conv1d_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv1d_weight', 'conv1d_bias']


class ModelNew(nn.Module):
    """
    Performs a standard 1D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0, dilation: int=1, groups: int=1, bias: bool=False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

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
    # State for conv1d (nn.Conv1d)
    if 'conv1d_weight' in flat_state:
        state_kwargs['conv1d_weight'] = flat_state['conv1d_weight']
    else:
        state_kwargs['conv1d_weight'] = getattr(model.conv1d, 'weight', None)
    if 'conv1d_bias' in flat_state:
        state_kwargs['conv1d_bias'] = flat_state['conv1d_bias']
    else:
        state_kwargs['conv1d_bias'] = getattr(model.conv1d, 'bias', None)
    state_kwargs['conv1d_stride'] = model.conv1d.stride
    state_kwargs['conv1d_padding'] = model.conv1d.padding
    state_kwargs['conv1d_dilation'] = model.conv1d.dilation
    state_kwargs['conv1d_groups'] = model.conv1d.groups
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

# -------------------------------------------------------------------------
# CUDA kernel source (optimized 1D convolution with shared memory tiling)
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_LENGTH 64
#define CHANNELS_PER_BLOCK 8

__global__ void conv1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int input_channels,
    int output_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    int input_tile_length
) {
    // Shared memory for input tile
    extern __shared__ float shared_input[];

    int output_pos = blockIdx.x * TILE_LENGTH + threadIdx.x;
    int output_channel = blockIdx.y * CHANNELS_PER_BLOCK + threadIdx.y;
    int batch_idx = blockIdx.z;

    if (output_channel >= output_channels || output_pos >= output_length)
        return;

    int group = output_channel / (output_channels / groups);
    int input_channel_start = group * (input_channels / groups);
    int channels_per_group = input_channels / groups;
    int local_output_channel = output_channel % (output_channels / groups);

    // Load input tile to shared memory
    int input_start = blockIdx.x * TILE_LENGTH * stride - padding;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;

    for (int i = tid; i < input_tile_length * channels_per_group; i += threads_per_block) {
        int ch = i / input_tile_length;
        int pos = i % input_tile_length;
        int input_pos = input_start + pos;
        int input_idx = batch_idx * input_channels * input_length +
                        (input_channel_start + ch) * input_length + input_pos;

        if (input_pos >= 0 && input_pos < input_length) {
            shared_input[ch * input_tile_length + pos] = input[input_idx];
        } else {
            shared_input[ch * input_tile_length + pos] = 0.0f;
        }
    }
    __syncthreads();

    // Perform convolution
    float sum = 0.0f;
    const float* weight_ptr = weight + (group * (output_channels / groups) + local_output_channel) * 
                              channels_per_group * kernel_size;

    for (int ic = 0; ic < channels_per_group; ++ic) {
        for (int k = 0; k < kernel_size; ++k) {
            int offset = threadIdx.x * stride + k * dilation;
            if (offset < input_tile_length) {
                sum += shared_input[ic * input_tile_length + offset] * 
                       weight_ptr[ic * kernel_size + k];
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[output_channel];
    }

    int output_idx = batch_idx * output_channels * output_length +
                     output_channel * output_length + output_pos;
    output[output_idx] = sum;
}

void conv1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    int batch_size = input.size(0);
    int input_channels = input.size(1);
    int input_length = input.size(2);
    int output_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto input_cont = input.contiguous();
    auto weight_cont = weight.contiguous();
    auto bias_cont = bias.contiguous();
    auto output_cont = output.contiguous();

    int tiles = (output_length + TILE_LENGTH - 1) / TILE_LENGTH;
    int channel_blocks = (output_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    
    dim3 grid(tiles, channel_blocks, batch_size);
    dim3 block(TILE_LENGTH, CHANNELS_PER_BLOCK);
    
    int input_tile_length = (TILE_LENGTH - 1) * stride + (kernel_size - 1) * dilation + 1;
    int shared_mem_size = input_tile_length * (input_channels / groups) * sizeof(float);

    conv1d_kernel<<<grid, block, shared_mem_size>>>(
        input_cont.data_ptr<float>(),
        weight_cont.data_ptr<float>(),
        bias_cont.defined() ? bias_cont.data_ptr<float>() : nullptr,
        output_cont.data_ptr<float>(),
        batch_size,
        input_channels,
        output_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        input_tile_length
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void conv1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d_forward", &conv1d_forward, "Optimized 1D Convolution Forward");
}
"""

# -------------------------------------------------------------------------
# Compile the CUDA extension
# -------------------------------------------------------------------------
conv_ext = load_inline(
    name='optimized_conv1d',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Model parameters
# -------------------------------------------------------------------------
batch_size = 32
in_channels = 64
out_channels = 128
kernel_size = 3
length = 131072

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
    return [x]

# -------------------------------------------------------------------------
# Optimized functional model
# -------------------------------------------------------------------------
def functional_model(
    x,
    *,
    conv1d_weight,
    conv1d_bias,
    conv1d_stride,
    conv1d_padding,
    conv1d_dilation,
    conv1d_groups,
):
    device = torch.device('cuda')
    x = x.to(device)
    conv1d_weight = conv1d_weight.to(device)
    if conv1d_bias is not None:
        conv1d_bias = conv1d_bias.to(device)
    else:
        conv1d_bias = torch.zeros(conv1d_weight.size(0), device=device, dtype=x.dtype)

    input_length = x.size(2)
    output_length = (input_length + 2 * conv1d_padding - 
                     conv1d_dilation * (kernel_size - 1) - 1) // conv1d_stride + 1
    
    output = torch.empty(
        (x.size(0), conv1d_weight.size(0), output_length),
        device=device,
        dtype=x.dtype
    )

    conv_ext.conv1d_forward(
        x, conv1d_weight, conv1d_bias, output,
        conv1d_stride, conv1d_padding, conv1d_dilation, conv1d_groups
    )

    return output
