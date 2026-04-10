# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152911/code_3.py
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

# ----------------------------------------------------------------------
# Parameters (only for reference – they are supplied by the harness)
# ----------------------------------------------------------------------
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
    return [torch.rand(batch_size, in_channels, height, width)]

# ----------------------------------------------------------------------
# Inline CUDA kernel that fuses convtranspose2d → add → ReLU → GELU → mul
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// ConvTranspose2d implementation adapted from CUTLASS-style kernels
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_ch, int in_h, int in_w,
    int out_ch, int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w,
    int dilation_h, int dilation_w,
    int groups) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_ch * (in_h * stride_h) * (in_w * stride_w);
    
    if (tid >= total_threads) return;
    
    int batch_idx = tid / (out_ch * (in_h * stride_h) * (in_w * stride_w));
    int temp = tid % (out_ch * (in_h * stride_h) * (in_w * stride_w));
    int out_ch_idx = temp / ((in_h * stride_h) * (in_w * stride_w));
    temp = temp % ((in_h * stride_h) * (in_w * stride_w));
    int out_y = temp / (in_w * stride_w);
    int out_x = temp % (in_w * stride_w);
    
    if (batch_idx >= batch_size || out_ch_idx >= out_ch) return;
    
    float val = 0.0f;
    bool valid = false;
    
    int group_id = out_ch_idx / (out_ch / groups);
    int group_out_ch_start = group_id * (out_ch / groups);
    int group_in_ch_start = group_id * (in_ch / groups);
    int group_in_ch_count = in_ch / groups;
    
    for (int ky = 0; ky < kernel_h; ++ky) {
        for (int kx = 0; kx < kernel_w; ++kx) {
            int in_y = out_y - ky * dilation_h + pad_h;
            int in_x = out_x - kx * dilation_w + pad_w;
            
            if (in_y % stride_h == 0 && in_x % stride_w == 0) {
                in_y /= stride_h;
                in_x /= stride_w;
                
                if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                    valid = true;
                    for (int ic = 0; ic < group_in_ch_count; ++ic) {
                        int in_ch_idx = group_in_ch_start + ic;
                        int weight_idx = ((out_ch_idx * in_ch) + in_ch_idx) * kernel_h * kernel_w +
                                        ky * kernel_w + kx;
                        int input_idx = ((batch_idx * in_ch) + in_ch_idx) * in_h * in_w +
                                       in_y * in_w + in_x;
                        
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    if (valid) {
        val += bias[out_ch_idx];
        output[tid] = val;
    } else {
        output[tid] = bias[out_ch_idx];
    }
}

// Fused pointwise operations kernel
__global__ void fused_pointwise_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int64_t total_elems,
                                       float add_value,
                                       float multiply_value) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;

    float x = input[idx] + add_value;          // add
    x = fmaxf(x, 0.0f);                       // ReLU

    // GELU – fast tanh approximation (matches PyTorch's CUDA kernel)
    float tanh_arg = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    float gelu = 0.5f * x * (1.0f + tanhf(tanh_arg));

    output[idx] = gelu * multiply_value;       // multiply
}

void fused_model_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w,
    int dilation_h, int dilation_w,
    int groups,
    float add_value,
    float multiply_value
) {
    auto batch_size = input.size(0);
    auto in_ch = input.size(1);
    auto in_h = input.size(2);
    auto in_w = input.size(3);
    auto out_ch = weight.size(1); // Note: for transposed conv, weight shape is (in_ch, out_ch/groups, kH, kW)
    
    // ConvTranspose2d
    const int threads = 256;
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + out_pad_h + 1;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + out_pad_w + 1;
    int total_elems = batch_size * out_ch * out_h * out_w;
    int blocks = (total_elems + threads - 1) / threads;
    
    conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_ch, in_h, in_w,
        out_ch, kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        out_pad_h, out_pad_w,
        dilation_h, dilation_w,
        groups
    );
    
    // Synchronize to ensure conv is complete before pointwise ops
    cudaDeviceSynchronize();
    
    // Fused pointwise operations
    blocks = (total_elems + threads - 1) / threads;
    fused_pointwise_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(), output.data_ptr<float>(),
        total_elems, add_value, multiply_value);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_model_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w,
    int dilation_h, int dilation_w,
    int groups,
    float add_value,
    float multiply_value
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_model_op", &fused_model_op, "Fused ConvTranspose2d + Add + ReLU + GELU + Mul");
}
"""

fused_ext = load_inline(
    name='fused_model',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional model required by the evaluation harness
# ----------------------------------------------------------------------
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
    # Calculate output dimensions
    in_h, in_w = x.shape[2], x.shape[3]
    k_h, k_w = conv_transpose_weight.shape[2], conv_transpose_weight.shape[3]
    s_h, s_w = conv_transpose_stride if isinstance(conv_transpose_stride, (tuple, list)) else (conv_transpose_stride, conv_transpose_stride)
    p_h, p_w = conv_transpose_padding if isinstance(conv_transpose_padding, (tuple, list)) else (conv_transpose_padding, conv_transpose_padding)
    op_h, op_w = conv_transpose_output_padding if isinstance(conv_transpose_output_padding, (tuple, list)) else (conv_transpose_output_padding, conv_transpose_output_padding)
    d_h, d_w = conv_transpose_dilation if isinstance(conv_transpose_dilation, (tuple, list)) else (conv_transpose_dilation, conv_transpose_dilation)
    
    out_h = (in_h - 1) * s_h - 2 * p_h + d_h * (k_h - 1) + op_h + 1
    out_w = (in_w - 1) * s_w - 2 * p_w + d_w * (k_w - 1) + op_w + 1
    
    # Create output tensor
    output = torch.empty(x.shape[0], conv_transpose_weight.shape[1], out_h, out_w, device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_model_op(
        x, conv_transpose_weight, conv_transpose_bias, output,
        k_h, k_w, s_h, s_w, p_h, p_w, op_h, op_w, d_h, d_w, conv_transpose_groups,
        add_value, multiply_value
    )
    
    return output
