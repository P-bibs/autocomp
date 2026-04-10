# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160018/code_2.py
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
from torch.utils.cpp_extension import load_inline

# The CUDA kernel uses float4 for vectorized memory access and a grid-stride loop.
# Implements a high-performance convolution transpose fused with activation in one kernel.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Vectorized fused kernel - ConvTranspose2d + Add + Min + GELU + Mul
__global__ void fused_conv_tr_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float add_val,
    float mul_val,
    int batch_size,
    int in_ch,
    int out_ch,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int k_size,
    int stride,
    int padding
) {
    int out_elements = out_ch * out_h * out_w;
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // Grid-stride loop processing 4 output elements per thread
    for (int base_idx = tid; base_idx < batch_size * out_elements; base_idx += blockDim.x * gridDim.x * 4) {
        float vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        
        // Compute indices for 4 elements
        int batch_idx = base_idx / out_elements;
        int out_pos = base_idx % out_elements;
        
        if (batch_idx >= batch_size) break;
        
        // For each of the 4 elements
        for (int elem = 0; elem < 4 && (base_idx + elem) < (batch_size * out_elements); elem++) {
            int idx = base_idx + elem;
            int b = idx / out_elements;
            int pos = idx % out_elements;
            
            if (b >= batch_size) break;
            
            int out_c = pos / (out_h * out_w);
            int out_hw = pos % (out_h * out_w);
            int out_y = out_hw / out_w;
            int out_x = out_hw % out_w;
            
            // Conv transpose logic
            float sum = 0.0f;
            for (int ic = 0; ic < in_ch; ic++) {
                for (int ky = 0; ky < k_size; ky++) {
                    for (int kx = 0; kx < k_size; kx++) {
                        int in_y = out_y - ky * stride + padding;
                        int in_x = out_x - kx * stride + padding;
                        
                        if (in_y >= 0 && in_y < in_h * stride && 
                            in_x >= 0 && in_x < in_w * stride && 
                            in_y % stride == 0 && in_x % stride == 0) {
                            
                            in_y /= stride;
                            in_x /= stride;
                            
                            int in_idx = b * (in_ch * in_h * in_w) + ic * (in_h * in_w) + in_y * in_w + in_x;
                            int w_idx = out_c * (in_ch * k_size * k_size) + ic * (k_size * k_size) + ky * k_size + kx;
                            
                            sum += input[in_idx] * weight[w_idx];
                        }
                    }
                }
            }
            
            vals[elem] = sum + bias[out_c];
        }
        
        // Apply fused activation to all 4 elements
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (base_idx + j < batch_size * out_elements) {
                float v = vals[j] + add_val;
                v = fminf(v, 0.0f);
                vals[j] = fast_gelu(v) * mul_val;
            }
        }
        
        // Write output using vectorized store when aligned
        if ((base_idx + 3) < batch_size * out_elements && (reinterpret_cast<uint64_t>(&output[base_idx]) % 16 == 0)) {
            float4 out_vec = make_float4(vals[0], vals[1], vals[2], vals[3]);
            reinterpret_cast<float4*>(&output[base_idx])[0] = out_vec;
        } else {
            // Fallback to scalar stores
            for (int j = 0; j < 4 && (base_idx + j) < batch_size * out_elements; j++) {
                output[base_idx + j] = vals[j];
            }
        }
    }
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float add_val,
    float mul_val,
    int in_ch,
    int out_ch,
    int k_size,
    int stride,
    int padding
) {
    int batch_size = input.size(0);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_h = output.size(2);
    int out_w = output.size(3);
    
    int threads = 256;
    int elements = batch_size * out_ch * out_h * out_w;
    int blocks = min((elements + threads * 4 - 1) / (threads * 4), 65535);
    
    fused_conv_tr_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        add_val,
        mul_val,
        batch_size,
        in_ch,
        out_ch,
        in_h,
        in_w,
        out_h,
        out_w,
        k_size,
        stride,
        padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float add_val,
    float mul_val,
    int in_ch,
    int out_ch,
    int k_size,
    int stride,
    int padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Vectorized fused convolution-transpose activation");
}
"""

fused_ext = load_inline(
    name='fused_op',
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
    add_value,
    multiply_value,
):
    # Validate assumptions - this implementation assumes specific values
    assert conv_transpose_groups == 1, "Only groups=1 supported"
    assert conv_transpose_dilation == (1, 1), "Only dilation=1 supported"
    assert isinstance(conv_transpose_stride, tuple) and len(conv_transpose_stride) == 2, "Stride must be a tuple"
    assert conv_transpose_stride[0] == conv_transpose_stride[1], "Only square strides supported"
    assert conv_transpose_padding[0] == conv_transpose_padding[1], "Only square padding supported"
    assert conv_transpose_output_padding == (0, 0), "Output padding not supported"
    
    stride_val = conv_transpose_stride[0]
    padding_val = conv_transpose_padding[0]
    k_size = conv_transpose_weight.shape[2]
    
    batch_size = x.shape[0]
    in_ch = x.shape[1]
    in_h = x.shape[2]
    in_w = x.shape[3]
    
    # Calculate output dimensions
    out_h = (in_h - 1) * stride_val - 2 * padding_val + k_size + conv_transpose_output_padding[0]
    out_w = (in_w - 1) * stride_val - 2 * padding_val + k_size + conv_transpose_output_padding[1]
    out_ch = conv_transpose_weight.shape[0]
    
    out = torch.empty((batch_size, out_ch, out_h, out_w), device='cuda')
    
    fused_ext.fused_op(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        out, 
        float(add_value), 
        float(multiply_value),
        in_ch,
        out_ch,
        k_size,
        stride_val,
        padding_val
    )
    
    return out

# Constants for the model
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
