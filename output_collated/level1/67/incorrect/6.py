# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_155306/code_7.py
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

# 1. CUDA Kernel implementation
# The kernel tiles input channels, processing large 1D sequences by caching 
# input segments in shared memory to minimize high-latency global memory fetches.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1d_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch, const int in_channels, const int out_channels,
    const int kernel_size, const int in_length, const int out_length,
    const int stride, const int padding, const int dilation, const int groups,
    const int tile_size, const int tile_len)
{
    // One block per (batch, output_channel)
    const int b = blockIdx.x / out_channels;
    const int oc = blockIdx.x % out_channels;

    const int out_channels_per_group = out_channels / groups;
    const int in_channels_per_group = in_channels / groups;
    const int group_id = oc / out_channels_per_group;
    const int ic_start = group_id * in_channels_per_group;

    extern __shared__ float shared_tile[];

    const int num_tiles = (out_length + tile_size - 1) / tile_size;

    for (int t = 0; t < num_tiles; ++t) {
        int out_start = t * tile_size;
        int in_start = out_start * stride - padding;

        // Load input tile into shared memory (Coalesced access)
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int p = threadIdx.x; p < tile_len; p += blockDim.x) {
                int in_pos = in_start + p;
                if (in_pos >= 0 && in_pos < in_length) {
                    shared_tile[ic * tile_len + p] = input[((size_t)b * in_channels + ic) * in_length + in_pos];
                } else {
                    shared_tile[ic * tile_len + p] = 0.0f;
                }
            }
        }
        __syncthreads();

        // Compute outputs for this tile
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            int out_pos = out_start + i;
            if (out_pos < out_length) {
                float sum = bias[oc];
                for (int ic_idx = 0; ic_idx < in_channels_per_group; ++ic_idx) {
                    int ic = ic_start + ic_idx;
                    for (int k = 0; k < kernel_size; ++k) {
                        float w = weight[((size_t)oc * in_channels_per_group + ic_idx) * kernel_size + k];
                        sum += w * shared_tile[ic * tile_len + i + k * dilation];
                    }
                }
                output[((size_t)b * out_channels + oc) * out_length + out_pos] = sum;
            }
        }
        __syncthreads();
    }
}

void conv1d_forward_cuda(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
    torch::Tensor& output, int stride, int padding, int dilation, int groups)
{
    const int batch = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int in_length = input.size(2);
    const int out_length = output.size(2);
    
    const int tile_size = 64;
    const int tile_len = tile_size + (kernel_size - 1) * dilation;
    const int threads = 128;
    const int blocks = batch * out_channels;
    const size_t shared_mem = in_channels * tile_len * sizeof(float);

    conv1d_tiled_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch, in_channels, out_channels, kernel_size,
        in_length, out_length, stride, padding, dilation, groups, tile_size, tile_len
    );
}
"""

cpp_source = """
void conv1d_forward_cuda(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& output, int stride, int padding, int dilation, int groups);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d", &conv1d_forward_cuda, "Tiled Conv1D");
}
"""

conv_ext = load_inline(name='conv_ext', cpp_sources=cpp_source, cuda_sources=cuda_source, extra_cuda_cflags=['-O3'])

def functional_model(x, *, conv1d_weight, conv1d_bias, conv1d_stride, conv1d_padding, conv1d_dilation, conv1d_groups):
    in_len = x.shape[2]
    kernel_size = conv1d_weight.shape[2]
    out_len = (in_len + 2 * conv1d_padding - conv1d_dilation * (kernel_size - 1) - 1) // conv1d_stride + 1
    out = torch.empty((x.shape[0], conv1d_weight.shape[0], out_len), device=x.device, dtype=x.dtype)
    conv_ext.conv1d(x.contiguous(), conv1d_weight.contiguous(), conv1d_bias, out, conv1d_stride, conv1d_padding, conv1d_dilation, conv1d_groups)
    return out

batch_size, in_channels, out_channels, kernel_size, length = 32, 64, 128, 3, 131072
def get_init_inputs(): return [in_channels, out_channels, kernel_size]
def get_inputs(): return [torch.rand(batch_size, in_channels, length)]
