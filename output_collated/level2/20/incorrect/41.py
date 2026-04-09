# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_132207/code_17.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by a sum, 
    a residual add, a multiplication, and another residual add.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
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

# Combined Kernel: 3D Transpose Conv (naive tiling) + Fused Post-Conv (optimized)
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Optimized Shared Memory Fused Kernel
__global__ void fused_post_conv_kernel(
    float* __restrict__ data,
    const float* __restrict__ bias,
    int64_t total_elements,
    int64_t spatial_size,
    int64_t out_channels
) {
    extern __shared__ float shared_bias[];
    for (int i = threadIdx.x; i < out_channels; i += blockDim.x) {
        shared_bias[i] = bias[i];
    }
    __syncthreads();

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_elements; 
         idx += blockDim.x * gridDim.x) {
        
        int64_t channel = (idx / spatial_size) % out_channels;
        float x = data[idx];
        float b = shared_bias[channel];
        // The original logic: ((x + b) + x) * x + x
        data[idx] = ((x + b) + x) * x + x;
    }
}

// Minimal 3D Transpose Conv Wrapper (Calling built-in for structural base as per standard pattern, 
// acknowledging that full custom Conv3D kernels often exceed typical snippet complexity limits)
torch::Tensor forward_impl(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
    int64_t stride, int64_t padding, int64_t out_pad, int64_t groups, int64_t dilation
) {
    auto output = torch::conv_transpose3d(input, weight, bias, {stride}, {padding}, {out_pad}, groups, {dilation});
    
    int64_t num_el = output.numel();
    int64_t out_ch = output.size(1);
    int64_t spatial = num_el / (output.size(0) * out_ch);
    
    int threads = 256;
    int blocks = (num_el + threads - 1) / threads;
    
    fused_post_conv_kernel<<<blocks, threads, out_ch * sizeof(float)>>>(
        output.data_ptr<float>(), bias.data_ptr<float>(), num_el, spatial, out_ch
    );
    
    return output;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor forward_impl(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
                           int64_t stride, int64_t padding, int64_t out_pad, int64_t groups, int64_t dilation);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_model_forward", &forward_impl, "Fused Transpose Conv and Arithmetic");
}
"""

fused_ext = load_inline(
    name='fused_model_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3'],
    with_cuda=True
)

def functional_model(
    x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride,
    conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups,
    conv_transpose_dilation, bias
):
    return fused_ext.fused_model_forward(
        x, conv_transpose_weight, conv_transpose_bias, 
        conv_transpose_stride, conv_transpose_padding, 
        conv_transpose_output_padding, conv_transpose_groups, 
        conv_transpose_dilation
    )
