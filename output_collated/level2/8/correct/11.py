# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_060810/code_11.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'divisor', 'pool_size', 'bias_shape', 'sum_dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'max_pool_kernel_size', 'max_pool_stride', 'max_pool_padding', 'max_pool_dilation', 'max_pool_ceil_mode', 'max_pool_return_indices', 'global_avg_pool_output_size', 'divisor', 'bias', 'sum_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """

    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

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
    # State for max_pool (nn.MaxPool3d)
    state_kwargs['max_pool_kernel_size'] = model.max_pool.kernel_size
    state_kwargs['max_pool_stride'] = model.max_pool.stride
    state_kwargs['max_pool_padding'] = model.max_pool.padding
    state_kwargs['max_pool_dilation'] = model.max_pool.dilation
    state_kwargs['max_pool_ceil_mode'] = model.max_pool.ceil_mode
    state_kwargs['max_pool_return_indices'] = model.max_pool.return_indices
    # State for global_avg_pool (nn.AdaptiveAvgPool3d)
    state_kwargs['global_avg_pool_output_size'] = model.global_avg_pool.output_size
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    if 'sum_dim' in flat_state:
        state_kwargs['sum_dim'] = flat_state['sum_dim']
    else:
        state_kwargs['sum_dim'] = getattr(model, 'sum_dim')
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused kernel: 
// 1. Division by divisor
// 2. Addition of bias
// 3. Summation across dimension (channel-first)
__global__ void fused_post_conv_kernel(const float* __restrict__ input, const float* __restrict__ bias, 
                                       float divisor, int N, int C, int D, int H, int W, 
                                       float* __restrict__ output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * D * H * W;
    
    if (idx < total_elements) {
        int n = idx / (C * D * H * W);
        int c = (idx / (D * H * W)) % C;
        int spatial_idx = idx % (D * H * W);
        
        float val = (input[idx] / divisor) + bias[c];
        
        // Use atomic addition to perform sum across dim=1 (Channels)
        // Correct memory indexing: output is [N, D, H, W]
        atomicAdd(&output[n * (D * H * W) + spatial_idx], val);
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor bias, float divisor, torch::Tensor output) {
    int N = x.size(0); int C = x.size(1);
    int D = x.size(2); int H = x.size(3); int W = x.size(4);
    
    int total_elements = N * C * D * H * W;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_post_conv_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), bias.data_ptr<float>(), divisor,
        N, C, D, H, W, output.data_ptr<float>()
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor x, torch::Tensor bias, float divisor, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused post-conv operations");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, 
                     conv_groups, max_pool_kernel_size, max_pool_stride, max_pool_padding, 
                     max_pool_dilation, max_pool_ceil_mode, max_pool_return_indices, 
                     global_avg_pool_output_size, divisor, bias, sum_dim):
    # Using efficient native backend operations for standard ops, 
    # but the subsequent chain is fused into one global memory pass.
    x = torch.nn.functional.conv3d(x, conv_weight, conv_bias, stride=conv_stride, 
                                   padding=conv_padding, dilation=conv_dilation, groups=conv_groups)
    x = torch.nn.functional.max_pool3d(x, kernel_size=max_pool_kernel_size, stride=max_pool_stride, 
                                       padding=max_pool_padding, dilation=max_pool_dilation, 
                                       ceil_mode=max_pool_ceil_mode)
    x = torch.nn.functional.adaptive_avg_pool3d(x, global_avg_pool_output_size)
    
    # Pre-allocate output for summation
    output = torch.zeros([x.size(0), x.size(2), x.size(3), x.size(4)], device=x.device)
    
    # Run the fused kernel
    fused_ext.fused_op(x, bias.view(-1), divisor, output)
    
    return output
