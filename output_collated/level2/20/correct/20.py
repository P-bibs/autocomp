# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_132207/code_13.py
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

# Optimized CUDA kernel using shared memory for bias and read-only cache loads
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void fused_post_conv_kernel(
    const float4* __restrict__ input,
    const float* __restrict__ bias,   // global bias (read-only)
    float4* __restrict__ output,
    int64_t num_elements_float4,
    int64_t spatial_size,
    int64_t out_channels
) {
    // ---- shared memory for bias ---------------------------------------
    extern __shared__ float bias_shared[];

    // ---- load bias once per block --------------------------------------
    if (threadIdx.x < out_channels) {
        bias_shared[threadIdx.x] = bias[threadIdx.x];
    }
    __syncthreads();

    // Grid-stride loop: each thread processes multiple float4 elements
    #pragma unroll 4
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < num_elements_float4;
         idx += blockDim.x * gridDim.x) {

        // ---- compute channel index for the first element in the group
        int64_t elem_idx   = idx * 4;                     // absolute element index
        int64_t base_ch    = (elem_idx / spatial_size) % out_channels;

        // ---- read input via the read-only cache ------------------------
        float4 x_vec = __ldg(&input[idx]);
        float4 result;

        // ---- element-wise arithmetic ----------------------------------
        float b = bias_shared[base_ch];   // use cached bias

        // The original formula: ((x + b) + x) * x + x
        // Re-written to let the compiler expose more FMA ops
        float x = x_vec.x;
        result.x = ( ( (x + b) + x ) * x ) + x;
        x = x_vec.y;
        result.y = ( ( (x + b) + x ) * x ) + x;
        x = x_vec.z;
        result.z = ( ( (x + b) + x ) * x ) + x;
        x = x_vec.w;
        result.w = ( ( (x + b) + x ) * x ) + x;

        output[idx] = result;
    }
}

void fused_post_conv_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output
) {
    const int64_t num_elements       = input.numel();
    const int64_t num_elements_float4 = (num_elements + 3) / 4;   // ceiling
    const int64_t spatial_size = input.size(2) * input.size(3) * input.size(4);
    const int64_t out_channels = input.size(1);

    const int threads_per_block = 256;
    const int blocks_per_sm     = 4;
    const int num_sms           = 68;               // RTX 2080 Ti
    const int blocks = min(num_sms * blocks_per_sm,
                           (int)((num_elements_float4 + threads_per_block - 1) / threads_per_block));

    // Shared memory size = out_channels * sizeof(float)
    const size_t shared_mem_bytes = out_channels * sizeof(float);

    const float4* input_ptr = reinterpret_cast<const float4*>(input.data_ptr<float>());
    float4* output_ptr = reinterpret_cast<float4*>(output.data_ptr<float>());

    fused_post_conv_kernel<<<blocks, threads_per_block, shared_mem_bytes>>>(
        input_ptr,
        bias.data_ptr<float>(),
        output_ptr,
        num_elements_float4,
        spatial_size,
        out_channels
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_post_conv_forward(const torch::Tensor& input,
                             const torch::Tensor& bias,
                             torch::Tensor& output);

torch::Tensor fused_post_conv(const torch::Tensor& input,
                              const torch::Tensor& bias) {
    auto output = torch::empty_like(input);
    fused_post_conv_forward(input, bias, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv", &fused_post_conv,
          "Fused post-conv arithmetic with shared-memory bias cache");
}
"""

# Compile the inline CUDA extension
fused_ext = load_inline(
    name='fused_post_conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
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
    # ---- Transposed convolution (kept unchanged – already optimal for the pattern)
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

    # Flatten bias for simple indexing
    bias_flat = bias.view(-1)

    # ---- Optimized fused kernel (bias now lives in shared memory)
    return fused_ext.fused_post_conv(x, bias_flat)

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
