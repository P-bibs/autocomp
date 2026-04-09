# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_130922/code_13.py
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

# Optimized CUDA kernel using shared memory for bias, float4 vectorization, and fast math
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_post_conv_kernel(
    const float4* __restrict__ input,
    const float* __restrict__ bias,
    float4* __restrict__ output,
    const int num_elements_float4,
    const int spatial_size,
    const int out_channels,
    const int total_elements
) {
    // Dynamic shared memory for the bias (size = out_channels * sizeof(float))
    extern __shared__ float bias_shared[];

    // ---- Load bias into shared memory -------------------------------------------------
    int tid = threadIdx.x;
    if (tid < out_channels) {
        bias_shared[tid] = bias[tid];
    }
    __syncthreads();

    // ---- Main loop --------------------------------------------------------------------
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements_float4) {
        int elem_base = idx * 4;                 // flat index of the first of the 4 elements
        // If the whole float4 is out of range, write zeros and exit
        if (elem_base >= total_elements) {
            output[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            return;
        }

        // Determine channel for this group (the first element decides the channel)
        int channel = (elem_base / spatial_size) % out_channels;
        float b = bias_shared[channel];          // bias from shared memory

        // Load the 4 packed floats
        float4 x_vec = input[idx];

        const float two = 2.0f;
        const float one = 1.0f;

        // Mask out components that lie outside the valid range
        float x0 = (elem_base < total_elements) ? x_vec.x : 0.0f;
        float x1 = (elem_base + 1 < total_elements) ? x_vec.y : 0.0f;
        float x2 = (elem_base + 2 < total_elements) ? x_vec.z : 0.0f;
        float x3 = (elem_base + 3 < total_elements) ? x_vec.w : 0.0f;

        // Compute the arithmetic: out = x * (2*x + bias + 1)
        float4 res;
        res.x = __fmul_rn(x0, __fadd_rn(__fmul_rn(two, x0), __fadd_rn(b, one)));
        res.y = __fmul_rn(x1, __fadd_rn(__fmul_rn(two, x1), __fadd_rn(b, one)));
        res.z = __fmul_rn(x2, __fadd_rn(__fmul_rn(two, x2), __fadd_rn(b, one)));
        res.w = __fmul_rn(x3, __fadd_rn(__fmul_rn(two, x3), __fadd_rn(b, one)));

        output[idx] = res;
    }
}

// Host function that launches the kernel
void fused_post_conv_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output
) {
    const int total_elements = static_cast<int>(input.numel());
    const int num_elements_float4 = (total_elements + 3) / 4;
    const int spatial_size = static_cast<int>(input.size(2) * input.size(3) * input.size(4));
    const int out_channels = static_cast<int>(input.size(1));

    const float4* input_ptr = reinterpret_cast<const float4*>(input.data_ptr<float>());
    float4* output_ptr = reinterpret_cast<float4*>(output.data_ptr<float>());

    const int threads = 256;
    const int blocks = (num_elements_float4 + threads - 1) / threads;

    const int shared_mem = out_channels * sizeof(float);

    fused_post_conv_kernel<<<blocks, threads, shared_mem>>>(
        input_ptr,
        bias.data_ptr<float>(),
        output_ptr,
        num_elements_float4,
        spatial_size,
        out_channels,
        total_elements
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
          "Fused post-convolution arithmetic with shared-memory bias and float4 vectorisation");
}
"""

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
    # Perform the convolution (as per the requested structure)
    x = F.conv_transpose3d(x, conv_transpose_weight, conv_transpose_bias, 
                          stride=conv_transpose_stride, padding=conv_transpose_padding, 
                          output_padding=conv_transpose_output_padding, 
                          groups=conv_transpose_groups, dilation=conv_transpose_dilation)
    
    # Flatten bias for simplified kernel indexing
    bias_flat = bias.view(-1)
    
    # Use optimized fused kernel for the intensive post-processing element-wise ops
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
