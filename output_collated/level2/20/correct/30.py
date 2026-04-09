# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_133559/code_20.py
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

# -------------------------------------------------------------------------
# CUDA Kernel: Fused Transposed Conv arithmetic using Grid-Stride
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_post_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int num_elements,
    const int spatial_size,
    const int out_channels)
{
    // Grid-stride loop for high occupancy and hardware utilization
    int elements_in_grid = blockDim.x * gridDim.x * 4;
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    for (int idx = tid; idx < num_elements; idx += elements_in_grid) {
        // Compute channel index: (idx / (D*H*W)) % C
        int channel_idx = (idx / spatial_size) % out_channels;
        float bias_val = __ldg(&bias[channel_idx]);
        
        float4 result_vec;
        float* res_ptr = reinterpret_cast<float*>(&result_vec);
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (idx + i < num_elements) {
                float x = __ldg(&input[idx + i]);
                // ((2*x + b) * x) + x
                res_ptr[i] = ((2.0f * x + bias_val) * x) + x;
            } else {
                res_ptr[i] = 0.0f;
            }
        }
        reinterpret_cast<float4*>(output + idx)[0] = result_vec;
    }
}

void fused_post_conv_launcher(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output)
{
    const int num_elements = static_cast<int>(input.numel());
    // spatial_size = D * H * W
    const int spatial_size = static_cast<int>(input.size(2) * input.size(3) * input.size(4));
    const int out_channels = static_cast<int>(input.size(1));

    const int threads = 256;
    // Launch based on compute capability / SM count typically, 1024 blocks is safe
    const int blocks = 1024;
    
    fused_post_conv_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        spatial_size,
        out_channels
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_post_conv_launcher(const torch::Tensor& input, const torch::Tensor& bias, torch::Tensor& output);

torch::Tensor fused_post_conv(const torch::Tensor& input, const torch::Tensor& bias) {
    auto output = torch::empty_like(input);
    fused_post_conv_launcher(input, bias, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv", &fused_post_conv, "Fused post-conv arithmetic with grid-stride");
}
"""

fused_ext = load_inline(
    name='fused_post_conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '--ptxas-options=-v'],
    with_cuda=True
)

def functional_model(
    x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride,
    conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups,
    conv_transpose_dilation, bias
):
    # Perform the Transposed Convolution
    # Note: Using native conv_transpose3d as the fundamental operator, 
    # but the subsequent post-processing is highly optimized via fused kernel.
    x = torch.nn.functional.conv_transpose3d(
        x, conv_transpose_weight, conv_transpose_bias,
        stride=conv_transpose_stride, padding=conv_transpose_padding,
        output_padding=conv_transpose_output_padding, groups=conv_transpose_groups,
        dilation=conv_transpose_dilation
    )

    # Execute the fused grid-stride kernel
    return fused_ext.fused_post_conv(x.contiguous(), bias.view(-1).contiguous())
