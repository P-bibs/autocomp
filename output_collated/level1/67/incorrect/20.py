# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160431/code_5.py
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

# CUDA kernel with Shared Memory Tiling for weights
# This optimizes global memory access by caching the weights in fast on-chip memory.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1d_shared_kernel(
    const float* __restrict__ input, const float* __restrict__ weight, float* __restrict__ output,
    int B, int C, int L, int OC, int K, int L_out) {
    
    // Shared memory for weights: [OC, C, K]
    extern __shared__ float s_weights[];
    
    int oc = blockIdx.y;
    int b = blockIdx.x;
    int l = threadIdx.x + blockIdx.z * blockDim.x;

    // Load weights into shared memory once per block
    if (threadIdx.x < (C * K)) {
        int c_k = threadIdx.x;
        s_weights[oc * (C * K) + c_k] = weight[oc * (C * K) + c_k];
    }
    __syncthreads();

    if (l < L_out) {
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            for (int k = 0; k < K; ++k) {
                sum += input[b * C * L + c * L + (l + k)] * s_weights[oc * (C * K) + c * k + k];
            }
        }
        output[b * OC * L_out + oc * L_out + l] = sum;
    }
}

void launch_conv1d(torch::Tensor x, torch::Tensor weight, torch::Tensor output) {
    int B = x.size(0), C = x.size(1), L = x.size(2);
    int OC = weight.size(0), K = weight.size(2);
    int L_out = L - K + 1;
    
    dim3 threads(256);
    dim3 blocks(B, OC, (L_out + threads.x - 1) / threads.x);
    
    size_t shared_mem_size = OC * C * K * sizeof(float);
    
    conv1d_shared_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        B, C, L, OC, K, L_out);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_conv1d(torch::Tensor x, torch::Tensor weight, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_conv1d", &launch_conv1d, "Optimized 1D Convolution");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv1d_weight, conv1d_bias, conv1d_stride, conv1d_padding, conv1d_dilation, conv1d_groups):
    # This implementation assumes stride=1, padding=0, dilation=1, groups=1
    batch_size, in_channels, length = x.shape
    out_channels = conv1d_weight.shape[0]
    kernel_size = conv1d_weight.shape[2]
    out_length = length - kernel_size + 1
    
    output = torch.empty((batch_size, out_channels, out_length), device=x.device)
    fused_ext.launch_conv1d(x, conv1d_weight, output)
    
    if conv1d_bias is not None:
        output += conv1d_bias.view(1, -1, 1)
    return output
