# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_133559/code_0.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------------------
# Optimized CUDA kernel – uses shared memory for bias caching
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
    // Shared memory for bias values
    extern __shared__ float shared_bias[];

    // Cooperative loading of bias into shared memory
    int tid = threadIdx.x;
    for (int i = tid; i < out_channels; i += blockDim.x) {
        shared_bias[i] = bias[i];
    }
    __syncthreads(); // Wait for all threads to load bias

    // Each thread handles 4 consecutive elements (vectorized load/store)
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx >= num_elements) return;

    // --------------------------------------------------------------
    // 1) Compute channel index once per thread (hoist division)
    // --------------------------------------------------------------
    int channel_idx = (idx / spatial_size) % out_channels;

    // --------------------------------------------------------------
    // 2) Load bias from shared memory
    // --------------------------------------------------------------
    float bias_val = shared_bias[channel_idx];

    // --------------------------------------------------------------
    // 3) Process the four elements
    // --------------------------------------------------------------
    float4 result_vec;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (idx + i < num_elements) {
            // Load input element via the read-only cache
            float x = __ldg(&input[idx + i]);

            // Compute ((x + bias) + x) * x + x  = 2*x*x + bias*x + x
            float tmp  = x + bias_val;          // x + b
            float tmp2 = tmp + x;               // (x + b) + x = 2*x + b
            float res  = tmp2 * x + x;          // ((2*x + b) * x) + x

            reinterpret_cast<float*>(&result_vec)[i] = res;
        } else {
            // Unused lane (won't be written because of the bound check)
            reinterpret_cast<float*>(&result_vec)[i] = 0.0f;
        }
    }

    // --------------------------------------------------------------
    // 4) Vectorized store
    // --------------------------------------------------------------
    reinterpret_cast<float4*>(output + idx)[0] = result_vec;
}

void fused_post_conv_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output)
{
    const int num_elements = static_cast<int>(input.numel());
    const int spatial_size = static_cast<int>(input.size(2) * input.size(3) * input.size(4));
    const int out_channels = static_cast<int>(input.size(1));

    const int threads_per_block = 256;
    const int blocks = (num_elements + threads_per_block * 4 - 1) / (threads_per_block * 4);

    // Calculate shared memory size for bias
    const int shared_mem_size = out_channels * sizeof(float);

    fused_post_conv_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        spatial_size,
        out_channels);
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_post_conv_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output);

torch::Tensor fused_post_conv(const torch::Tensor& input, const torch::Tensor& bias) {
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    auto output = torch::empty_like(input);
    fused_post_conv_forward(input, bias, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv", &fused_post_conv, "Fused post-conv arithmetic");
}
"""

# -------------------------------------------------------------------------
# Build the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_post_conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional model used for evaluation
# -------------------------------------------------------------------------
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
    # ----- Transposed convolution (unchanged) -----
    x = F.conv_transpose3d(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        stride=conv_transpose_stride,
        padding=conv_transpose_padding,
        output_padding=conv_transpose_output_padding,
        groups=conv_transpose_groups,
        dilation=conv_transpose_dilation
    )

    # Flatten bias to a 1-D tensor (required by the kernel)
    bias_flat = bias.view(-1)

    # ----- Optimized fused kernel -----
    return fused_ext.fused_post_conv(x, bias_flat)

# -------------------------------------------------------------------------
# Helper code (shape parameters, input factories)
# -------------------------------------------------------------------------
batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
