# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_132207/code_12.py
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
# Optimized CUDA kernel – shared-memory bias + block-level channel index
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
    const int out_channels,
    const int channel_blocks_per_batch)
{
    // each thread handles a vector of 4 consecutive elements
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx >= num_elements) return;

    // ------------------------------------------------------------------
    // 1) Compute channel index once per block (no per-thread division)
    // ------------------------------------------------------------------
    int channel = (blockIdx.x / channel_blocks_per_batch) % out_channels;

    // ------------------------------------------------------------------
    // 2) Load bias once per block into shared memory
    // ------------------------------------------------------------------
    __shared__ float bias_val;
    if (threadIdx.x == 0) {
        bias_val = __ldg(&bias[channel]);
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // 3) Determine how many of the 4 lanes are valid
    // ------------------------------------------------------------------
    int remaining = num_elements - idx;
    int limit = (remaining >= 4) ? 4 : remaining;

    // ------------------------------------------------------------------
    // 4) Vectorized load of the input (full float4 when possible)
    // ------------------------------------------------------------------
    float4 in_vec;
    if (limit == 4) {
        // fully inside the tensor – use a coalesced vector load
        in_vec = reinterpret_cast<const float4*>(input)[idx / 4];
    } else {
        // partial vector – load element-wise to avoid out-of-bounds reads
        float tmp[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            tmp[i] = (i < remaining) ? __ldg(&input[idx + i]) : 0.0f;
        }
        reinterpret_cast<float*>(&in_vec)[0] = tmp[0];
        reinterpret_cast<float*>(&in_vec)[1] = tmp[1];
        reinterpret_cast<float*>(&in_vec)[2] = tmp[2];
        reinterpret_cast<float*>(&in_vec)[3] = tmp[3];
    }

    // ------------------------------------------------------------------
    // 5) Compute result:  x * (2*x + bias + 1)  using FMA
    // ------------------------------------------------------------------
    float4 out_vec;
    float b = bias_val;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (i < limit) {
            float x = reinterpret_cast<float*>(&in_vec)[i];
            float t = fma(x, 2.0f, b + 1.0f);   // t = 2*x + (b+1)
            float res = x * t;                  // res = x * t
            reinterpret_cast<float*>(&out_vec)[i] = res;
        } else {
            reinterpret_cast<float*>(&out_vec)[i] = 0.0f;
        }
    }

    // ------------------------------------------------------------------
    // 6) Vectorized store
    // ------------------------------------------------------------------
    reinterpret_cast<float4*>(output + idx)[0] = out_vec;
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
    const int block_elements   = threads_per_block * 4;                // 1024
    const int blocks           = (num_elements + block_elements - 1) / block_elements;

    // Number of blocks that belong to the same channel (spatial_size / block_elements)
    const int channel_blocks_per_batch = spatial_size / block_elements;

    fused_post_conv_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        spatial_size,
        out_channels,
        channel_blocks_per_batch);
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

    # ----- Optimized fused kernel (shared-memory bias, block-level channel) -----
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
