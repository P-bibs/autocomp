# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_070827/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv1d_weight', 'conv1d_bias', 'conv1d_stride', 'conv1d_padding', 'conv1d_dilation', 'conv1d_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv1d_weight', 'conv1d_bias']


class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool=False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

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
    # State for conv1d (nn.Conv2d)
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

# The custom CUDA implementation of a 2D convolution kernel (for stride=1, padding=0, dilation=1)
# Implements direct convolution with global memory coalescing and minimal launch overhead.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* __restrict__ input, 
                              const float* __restrict__ weight, 
                              const float* __restrict__ bias, 
                              float* __restrict__ output,
                              int B, int C, int H, int W, int OC, int K, int OH, int OW) {
    // Map thread/block indices to output spatial location (ty, tx) and batch/channel (n, oc)
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;

    if (tx < OW && ty < OH) {
        for (int b = 0; b < B; ++b) {
            float val = bias[oc];
            for (int c = 0; c < C; ++c) {
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        val += input[((b * C + c) * H + (ty + kh)) * W + (tx + kw)] * 
                               weight[((oc * C + c) * K + kh) * K + kw];
                    }
                }
            }
            output[((b * (blockDim.z * gridDim.z) + oc) * OH + ty) * OW + tx] = val;
        }
    }
}

void conv2d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, 
                    int B, int C, int H, int W, int OC, int K) {
    int OH = H - K + 1;
    int OW = W - K + 1;
    
    // Grid dimensions: (OW/16, OH/16, OC)
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((OW + 15) / 16, (OH + 15) / 16, OC);
    
    conv2d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        output.data_ptr<float>(), B, C, H, W, OC, K, OH, OW
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void conv2d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, 
                    int B, int C, int H, int W, int OC, int K);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_forward", &conv2d_forward, "Custom 2D Convolution");
}
"""

module = load_inline(
    name='custom_conv2d',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

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
    # Functional model implementation using the custom CUDA kernel
    B, C, H, W = x.shape
    OC, _, K, _ = conv1d_weight.shape
    OH, OW = H - K + 1, W - K + 1
    
    output = torch.empty((B, OC, OH, OW), device=x.device, dtype=x.dtype)
    
    module.conv2d_forward(x, conv1d_weight, conv1d_bias, output, B, C, H, W, OC, K)
    
    return output
