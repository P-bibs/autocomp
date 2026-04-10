# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_155151/code_6.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'add_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'add_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

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
    # State for conv_transpose (nn.ConvTranspose2d)
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
    if 'add_value' in flat_state:
        state_kwargs['add_value'] = flat_state['add_value']
    else:
        state_kwargs['add_value'] = getattr(model, 'add_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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

# Define the vectorized fused kernel for element-wise operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Fast GELU approximation (matches common hardware implementations)
__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Vectorized GELU for float4
__device__ __forceinline__ float4 fast_gelu_vec(float4 v) {
    return make_float4(fast_gelu(v.x),
                       fast_gelu(v.y),
                       fast_gelu(v.z),
                       fast_gelu(v.w));
}

// Vectorized kernel: processes 4 elements per thread
__global__ void fused_op_vector_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       const float add_val,
                                       const float mul_val,
                                       const int num_elements) {
    const int vec_size = 4;
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * vec_size;
    if (base >= num_elements) return;

    // Load up to 4 elements
    float4 val;
    if (base + vec_size <= num_elements) {
        // Full vector load (aligned)
        val = __ldg((const float4*)(input + base));
    } else {
        // Tail handling – load scalar and fill with zeros (will be discarded)
        float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = 0; i < vec_size && base + i < num_elements; ++i) {
            tmp[i] = __ldg(input + base + i);
        }
        val = make_float4(tmp[0], tmp[1], tmp[2], tmp[3]);
    }

    // Fused operations (component-wise): add → min(0) → GELU → mul
    // Add
    float4 v = make_float4(val.x + add_val,
                           val.y + add_val,
                           val.z + add_val,
                           val.w + add_val);

    // Min with zero
    v = make_float4(fminf(v.x, 0.0f),
                    fminf(v.y, 0.0f),
                    fminf(v.z, 0.0f),
                    fminf(v.w, 0.0f));

    // Apply fast GELU
    v = fast_gelu_vec(v);

    // Multiply
    v = make_float4(v.x * mul_val,
                    v.y * mul_val,
                    v.z * mul_val,
                    v.w * mul_val);

    // Store the result
    if (base + vec_size <= num_elements) {
        ((float4*)output)[base / vec_size] = v;
    } else {
        for (int i = 0; i < vec_size && base + i < num_elements; ++i) {
            output[base + i] = ((float*)&v)[i];
        }
    }
}

// Host function to launch the vectorized kernel
void fused_op_forward(torch::Tensor input,
                      torch::Tensor output,
                      float add_val,
                      float mul_val) {
    const int num_elements = input.numel();
    const int vec_size = 4;
    const int threads = 256;
    const int blocks = (num_elements + threads * vec_size - 1) / (threads * vec_size);

    fused_op_vector_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        add_val,
        mul_val,
        num_elements);
}
"""

# Define the C++ binding using pybind11
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor input,
                      torch::Tensor output,
                      float add_val,
                      float mul_val);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward,
          "Vectorized fused add-min-GELU-mul operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Define the functional model that uses the custom vectorized kernel
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
    add_value,
    multiply_value,
):
    # Perform transposed convolution manually using im2col + GEMM-like operation
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_h, kernel_w = conv_transpose_weight.shape
    
    # Calculate output dimensions
    out_height = (in_height - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + kernel_h + conv_transpose_output_padding[0]
    out_width = (in_width - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + kernel_w + conv_transpose_output_padding[1]
    
    # Manual implementation of conv_transpose2d (basic version)
    # Create an output tensor initialized to zero
    out = torch.zeros(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Loop over batches and apply kernel
    for b in range(batch_size):
        for c_out in range(out_channels):
            for h in range(in_height):
                for w in range(in_width):
                    # Apply kernel at each position
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            out_h = h * conv_transpose_stride[0] - conv_transpose_padding[0] + kh
                            out_w = w * conv_transpose_stride[1] - conv_transpose_padding[1] + kw
                            
                            if 0 <= out_h < out_height and 0 <= out_w < out_width:
                                val = x[b, :, h, w].view(-1)
                                weight = conv_transpose_weight[c_out, :, kh, kw].view(-1)
                                prod = (val * weight).sum()
                                if conv_transpose_bias is not None:
                                    prod += conv_transpose_bias[c_out]
                                out[b, c_out, out_h, out_w] += prod
    
    # Apply the vectorized fused operation (add → min(0) → GELU → mul)
    result = torch.empty_like(out)
    fused_ext.fused_op(out, result, float(add_value), float(multiply_value))
    return result

# Constants matching the original script
batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
