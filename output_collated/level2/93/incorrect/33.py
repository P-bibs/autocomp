# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_154319/code_4.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Optimized fused kernel with vectorized memory access
__global__ void fused_op_kernel_optimized(const float* __restrict__ input, 
                                          float* __restrict__ output, 
                                          float add_val, float mul_val, 
                                          int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process 4 elements per thread for better memory bandwidth utilization
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int element_idx = idx * 4 + i;
        if (element_idx < num_elements) {
            float val = input[element_idx] + add_val;
            val = fminf(val, 0.0f);
            val = fast_gelu(val);
            output[element_idx] = val * mul_val;
        }
    }
}

// Custom optimized conv_transpose2d kernel using implicit GEMM approach
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int KH, int KW,
    int stride, int padding, int output_padding, int groups, int dilation,
    int H_out, int W_out) {
    
    // Calculate output indices
    int batch_idx = blockIdx.z;
    int out_ch = blockIdx.y;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    
    // Shared memory for partial sums
    extern __shared__ float shared_mem[];
    float* partial_sums = shared_mem;
    
    // Each thread processes multiple output pixels
    int total_output_pixels = H_out * W_out;
    for (int pixel_idx = tid; pixel_idx < total_output_pixels; pixel_idx += total_threads) {
        int h_out = pixel_idx / W_out;
        int w_out = pixel_idx % W_out;
        
        float sum = (bias != nullptr) ? bias[out_ch] : 0.0f;
        
        // Determine which group this output channel belongs to
        int group_id = out_ch / (C_out / groups);
        int in_ch_start = group_id * (C_in / groups);
        int in_ch_end = in_ch_start + (C_in / groups);
        
        // Iterate through input channels in this group
        for (int in_ch = in_ch_start; in_ch < in_ch_end; ++in_ch) {
            // Iterate through kernel positions
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    // Compute corresponding input position
                    int h_in = (h_out + padding - kh * dilation) / stride;
                    int w_in = (w_out + padding - kw * dilation) / stride;
                    
                    // Check if input position is valid
                    if ((h_out + padding - kh * dilation) % stride == 0 &&
                        (w_out + padding - kw * dilation) % stride == 0 &&
                        h_in >= 0 && h_in < H_in &&
                        w_in >= 0 && w_in < W_in) {
                        
                        // Load input value
                        int input_idx = batch_idx * (C_in * H_in * W_in) +
                                        in_ch * (H_in * W_in) +
                                        h_in * W_in + w_in;
                        
                        // Load weight value
                        int weight_idx = ((in_ch - in_ch_start) * (C_out / groups) + 
                                         (out_ch % (C_out / groups))) * (KH * KW) +
                                         kh * KW + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Write result
        int output_idx = batch_idx * (C_out * H_out * W_out) +
                         out_ch * (H_out * W_out) +
                         h_out * W_out + w_out;
        output[output_idx] = sum;
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor output, float add_val, float mul_val) {
    int num_elements = input.numel();
    int threads = 256;
    // Process 4 elements per thread, so divide by 4
    int blocks = (num_elements + threads * 4 - 1) / (threads * 4);
    fused_op_kernel_optimized<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        add_val, mul_val, num_elements
    );
}

void conv_transpose2d_forward(torch::Tensor input, torch::Tensor weight, 
                              torch::Tensor bias, torch::Tensor output,
                              int stride, int padding, int output_padding,
                              int groups, int dilation) {
    auto input_shape = input.sizes();
    auto weight_shape = weight.sizes();
    auto output_shape = output.sizes();
    
    int N = input_shape[0];
    int C_in = input_shape[1];
    int H_in = input_shape[2];
    int W_in = input_shape[3];
    int C_out = weight_shape[1] * groups;
    int KH = weight_shape[2];
    int KW = weight_shape[3];
    int H_out = output_shape[2];
    int W_out = output_shape[3];
    
    // Launch configuration
    int threads_per_block = 256;
    int total_pixels = H_out * W_out;
    int shared_mem_size = threads_per_block * sizeof(float);
    
    dim3 blocks(1, C_out, N);
    dim3 threads(threads_per_block);
    
    conv_transpose2d_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, KH, KW,
        stride, padding, output_padding, groups, dilation,
        H_out, W_out
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor input, torch::Tensor output, float add_val, float mul_val);
void conv_transpose2d_forward(torch::Tensor input, torch::Tensor weight, 
                              torch::Tensor bias, torch::Tensor output,
                              int stride, int padding, int output_padding,
                              int groups, int dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused add-min-gelu-mul operation");
    m.def("conv_transpose2d", &conv_transpose2d_forward, "Custom conv_transpose2d");
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
    # Compute output dimensions for conv_transpose2d
    N, C_in, H_in, W_in = x.shape
    _, C_out_per_group, KH, KW = conv_transpose_weight.shape
    C_out = C_out_per_group * conv_transpose_groups
    
    H_out = (H_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + KH + conv_transpose_output_padding
    W_out = (W_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + KW + conv_transpose_output_padding
    
    # Custom conv_transpose2d
    conv_out = torch.empty(N, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
    fused_ext.conv_transpose2d(
        x, conv_transpose_weight, conv_transpose_bias, conv_out,
        conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding,
        conv_transpose_groups, conv_transpose_dilation
    )
    
    # Fused operation
    fused_out = torch.empty_like(conv_out)
    fused_ext.fused_op(conv_out, fused_out, float(add_value), float(multiply_value))
    
    return fused_out

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
