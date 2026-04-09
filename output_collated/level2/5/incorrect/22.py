# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_114641/code_4.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'bias_shape', 'stride', 'padding', 'output_padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
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

# CUDA Kernel: Custom ConvTranspose2d + Fused Bias Subtraction and Tanh
# Based on Implicit GEMM with vectorized memory access
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

// Device constants
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 256
#define ELEMENTS_PER_THREAD 4

// Utility: Ceil division
inline __host__ __device__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// Kernel 1: Im2Col transformation for ConvTranspose2d
__global__ void im2col_kernel(
    const float* __restrict__ data_im,
    float* __restrict__ data_col,
    int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int height_col, int width_col
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int num_kernels = channels * height_col * width_col * kernel_h * kernel_w;

    if (index >= num_kernels) return;

    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    index /= height_col;
    int channel_in = index % channels;
    index /= channels;
    int kernel_row = index % kernel_h;
    int kernel_col = index / kernel_h;

    int h_in = h_out * stride_h - pad_h + kernel_row;
    int w_in = w_out * stride_w - pad_w + kernel_col;

    data_col[(channel_in * kernel_h * kernel_w + kernel_row * kernel_w + kernel_col) * height_col * width_col + h_out * width_col + w_out] =
        (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) ?
        data_im[(channel_in * height + h_in) * width + w_in] : 0.0f;
}

// Kernel 2: Fused operation (Bias subtraction and Tanh) with vectorization
__global__ void fused_op_kernel(float4* __restrict__ data, const float* __restrict__ bias, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_vectors = (num_elements + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;

    if (idx < total_vectors) {
        float4 val = data[idx];
        int channel_base = (idx * ELEMENTS_PER_THREAD) % gridDim.y;
        if (idx * ELEMENTS_PER_THREAD + 3 < num_elements) {
            val.x = tanhf(val.x - bias[channel_base]);
            val.y = tanhf(val.y - bias[channel_base]);
            val.z = tanhf(val.z - bias[channel_base]);
            val.w = tanhf(val.w - bias[channel_base]);
        } else {
            if (idx * ELEMENTS_PER_THREAD + 0 < num_elements) val.x = tanhf(val.x - bias[channel_base]);
            if (idx * ELEMENTS_PER_THREAD + 1 < num_elements) val.y = tanhf(val.y - bias[channel_base]);
            if (idx * ELEMENTS_PER_THREAD + 2 < num_elements) val.z = tanhf(val.z - bias[channel_base]);
            if (idx * ELEMENTS_PER_THREAD + 3 < num_elements) val.w = tanhf(val.w - bias[channel_base]);
        }
        data[idx] = val;
    }
}

// Host function orchestrating the complete custom conv + fusion
void custom_conv_fusion_forward(
    torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w,
    int dilation_h, int dilation_w,
    int groups
) {
    // Conv dimensions
    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int in_group_channels = in_channels / groups;
    int out_group_channels = out_channels / groups;
    
    // Output dimensions calculation for transposed convolution
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + out_pad_h + 1;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + out_pad_w + 1;

    // Im2Col dimensions
    int height_col = (out_h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col = (out_w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int num_kernels = out_channels * kernel_h * kernel_w * height_col * width_col;

    // Create intermediate col buffer (columns)
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor col = torch::zeros({out_group_channels * kernel_h * kernel_w, height_col * width_col * batch}, options);

    // Resize output tensor
    output.resize_({batch, out_channels, out_h, out_w});

    // Process each group
    for (int g = 0; g < groups; ++g) {
        // 1. Im2Col kernel launch
        int threads = MAX_THREADS_PER_BLOCK;
        int blocks = div_up(num_kernels, threads);
        
        im2col_kernel<<<blocks, threads>>>(
            input.data_ptr<float>() + g * in_group_channels * in_h * in_w,
            col.data_ptr<float>(),
            in_group_channels, in_h, in_w,
            kernel_h, kernel_w,
            pad_h, pad_w,
            stride_h, stride_w,
            height_col, width_col
        );

        // 2. GEMM: (weight_T @ col) -> output slice
        cublasHandle_t cublas_handle;
        cublasCreate(&cublas_handle);
        
        const float *weight_ptr = weight.data_ptr<float>() + g * out_group_channels * in_group_channels * kernel_h * kernel_w;
        const float *col_ptr = col.data_ptr<float>();
        float *out_ptr = output.data_ptr<float>() + g * out_group_channels * out_h * out_w;
        
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(
            cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            out_group_channels, height_col * width_col * batch, in_group_channels * kernel_h * kernel_w,
            &alpha,
            weight_ptr, in_group_channels * kernel_h * kernel_w,
            col_ptr, in_group_channels * kernel_h * kernel_w,
            &beta,
            out_ptr, out_group_channels
        );
        
        cublasDestroy(cublas_handle);
    }

    // 3. Add convolution bias
    if (conv_bias.defined()) {
        auto bias_broadcast = conv_bias.view({1, out_channels, 1, 1});
        output.add_(bias_broadcast);
    }

    // 4. Launch fused kernel for bias subtraction and tanh
    int total_elements = batch * out_channels * out_h * out_w;
    int threads_fuse = MAX_THREADS_PER_BLOCK;
    int blocks_fuse = div_up(total_elements, threads_fuse * ELEMENTS_PER_THREAD);
    
    dim3 grid(blocks_fuse, out_channels); // Pass channels for proper indexing
    fused_op_kernel<<<grid, threads_fuse>>>(
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        post_bias.data_ptr<float>(),
        total_elements
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void custom_conv_fusion_forward(
    torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w,
    int dilation_h, int dilation_w,
    int groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &custom_conv_fusion_forward, "Custom Conv + Fusion Forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='conv_fusion_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-lcublas'],
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
    # Prepare output tensor
    output = torch.empty(0, device=x.device, dtype=x.dtype)
    
    # Call custom implementation
    fused_ext.forward(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        bias,
        output,
        conv_transpose_stride[0], conv_transpose_stride[1],
        conv_transpose_padding[0], conv_transpose_padding[1],
        conv_transpose_output_padding[0], conv_transpose_output_padding[1],
        conv_transpose_dilation[0], conv_transpose_dilation[1],
        conv_transpose_groups
    )
    
    return output

# Setup for testing
batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]
