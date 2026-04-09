# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_055208/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'divisor', 'pool_size', 'bias_shape', 'sum_dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'max_pool_kernel_size', 'max_pool_stride', 'max_pool_padding', 'max_pool_dilation', 'max_pool_ceil_mode', 'max_pool_return_indices', 'global_avg_pool_output_size', 'divisor', 'bias', 'sum_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """

    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

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
    # State for conv (nn.Conv3d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
    # State for max_pool (nn.MaxPool3d)
    state_kwargs['max_pool_kernel_size'] = model.max_pool.kernel_size
    state_kwargs['max_pool_stride'] = model.max_pool.stride
    state_kwargs['max_pool_padding'] = model.max_pool.padding
    state_kwargs['max_pool_dilation'] = model.max_pool.dilation
    state_kwargs['max_pool_ceil_mode'] = model.max_pool.ceil_mode
    state_kwargs['max_pool_return_indices'] = model.max_pool.return_indices
    # State for global_avg_pool (nn.AdaptiveAvgPool3d)
    state_kwargs['global_avg_pool_output_size'] = model.global_avg_pool.output_size
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    if 'sum_dim' in flat_state:
        state_kwargs['sum_dim'] = flat_state['sum_dim']
    else:
        state_kwargs['sum_dim'] = getattr(model, 'sum_dim')
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Load the custom CUDA kernel
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t>
__global__ void fused_conv_fused_op_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ conv_weight,
    const scalar_t* __restrict__ conv_bias,
    const scalar_t divisor,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int conv_out_depth,
    const int conv_out_height,
    const int conv_out_width,
    const int pool_kernel_d,
    const int pool_kernel_h,
    const int pool_kernel_w,
    const int pool_stride_d,
    const int pool_stride_h,
    const int pool_stride_w,
    const int pool_padding_d,
    const int pool_padding_h,
    const int pool_padding_w,
    const int pool_out_depth,
    const int pool_out_height,
    const int pool_out_width,
    const int global_avg_out_depth,
    const int global_avg_out_height,
    const int global_avg_out_width,
    const int sum_dim
) {
    // Convolution + other operations fused kernel
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * global_avg_out_depth * global_avg_out_height * global_avg_out_width;
    
    if (tid >= total_elements) return;

    // Calculate indices
    int temp_tid = tid;
    const int out_w_idx = temp_tid % global_avg_out_width;
    temp_tid /= global_avg_out_width;
    const int out_h_idx = temp_tid % global_avg_out_height;
    temp_tid /= global_avg_out_height;
    const int out_d_idx = temp_tid % global_avg_out_depth;
    temp_tid /= global_avg_out_depth;
    const int out_c_idx = temp_tid % out_channels;
    const int batch_idx = temp_tid / out_channels;

    // Map to pool output indices
    const int pool_out_w_start = out_w_idx * (pool_out_width / global_avg_out_width);
    const int pool_out_h_start = out_h_idx * (pool_out_height / global_avg_out_height);
    const int pool_out_d_start = out_d_idx * (pool_out_depth / global_avg_out_depth);
    
    const int pool_out_w_end = (out_w_idx + 1) * (pool_out_width / global_avg_out_width);
    const int pool_out_h_end = (out_h_idx + 1) * (pool_out_height / global_avg_out_height);
    const int pool_out_d_end = (out_d_idx + 1) * (pool_out_depth / global_avg_out_depth);

    // Adaptive average pooling calculation
    scalar_t sum_val = 0.0;
    int count = 0;
    
    for (int pd = pool_out_d_start; pd < pool_out_d_end; pd++) {
        for (int ph = pool_out_h_start; ph < pool_out_h_end; ph++) {
            for (int pw = pool_out_w_start; pw < pool_out_w_end; pw++) {
                // Map to conv output indices
                const int conv_w_start = pw * pool_stride_w - pool_padding_w;
                const int conv_h_start = ph * pool_stride_h - pool_padding_h;
                const int conv_d_start = pd * pool_stride_d - pool_padding_d;
                
                // Max pooling within the pooling window
                scalar_t max_val = -1e9;
                for (int kd = 0; kd < pool_kernel_d; kd++) {
                    for (int kh = 0; kh < pool_kernel_h; kh++) {
                        for (int kw = 0; kw < pool_kernel_w; kw++) {
                            const int conv_d = conv_d_start + kd;
                            const int conv_h = conv_h_start + kh;
                            const int conv_w = conv_w_start + kw;
                            
                            if (conv_d >= 0 && conv_d < conv_out_depth &&
                                conv_h >= 0 && conv_h < conv_out_height &&
                                conv_w >= 0 && conv_w < conv_out_width) {
                                
                                // Convolution calculation for this point
                                scalar_t conv_val = conv_bias[out_c_idx];
                                for (int ic = 0; ic < in_channels; ic++) {
                                    for (int kd2 = 0; kd2 < kernel_d; kd2++) {
                                        for (int kh2 = 0; kh2 < kernel_h; kh2++) {
                                            for (int kw2 = 0; kw2 < kernel_w; kw2++) {
                                                const int in_d = conv_d * stride_d + kd2 - padding_d;
                                                const int in_h = conv_h * stride_h + kh2 - padding_h;
                                                const int in_w = conv_w * stride_w + kw2 - padding_w;
                                                
                                                if (in_d >= 0 && in_d < input_depth &&
                                                    in_h >= 0 && in_h < input_height &&
                                                    in_w >= 0 && in_w < input_width) {
                                                    
                                                    const int in_idx = batch_idx * in_channels * input_depth * input_height * input_width +
                                                                       ic * input_depth * input_height * input_width +
                                                                       in_d * input_height * input_width +
                                                                       in_h * input_width +
                                                                       in_w;
                                                    const int weight_idx = out_c_idx * in_channels * kernel_d * kernel_h * kernel_w +
                                                                           ic * kernel_d * kernel_h * kernel_w +
                                                                           kd2 * kernel_h * kernel_w +
                                                                           kh2 * kernel_w +
                                                                           kw2;
                                                    conv_val += input[in_idx] * conv_weight[weight_idx];
                                                }
                                            }
                                        }
                                    }
                                }
                                
                                const scalar_t pooled_val = conv_val / divisor;
                                if (pooled_val > max_val) {
                                    max_val = pooled_val;
                                }
                            }
                        }
                    }
                }
                
                if (max_val != -1e9) {
                    sum_val += max_val;
                    count++;
                }
            }
        }
    }
    
    if (count > 0) {
        const scalar_t avg_val = sum_val / count;
        output[tid] = avg_val + bias[out_c_idx];
    } else {
        output[tid] = bias[out_c_idx];
    }
}

// Reduction kernel for sum operation
template <typename scalar_t>
__global__ void sum_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int reduction_size,
    const int output_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_size) return;
    
    scalar_t sum = 0.0;
    for (int i = 0; i < reduction_size; i++) {
        sum += input[tid + i * output_size];
    }
    output[tid] = sum;
}

void fused_conv_fused_op_forward(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    float divisor,
    torch::Tensor bias,
    torch::Tensor output,
    int out_channels,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int pool_kernel_d, int pool_kernel_h, int pool_kernel_w,
    int pool_stride_d, int pool_stride_h, int pool_stride_w,
    int pool_padding_d, int pool_padding_h, int pool_padding_w,
    int global_avg_out_depth, int global_avg_out_height, int global_avg_out_width,
    int sum_dim
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_depth = input.size(2);
    const int input_height = input.size(3);
    const int input_width = input.size(4);
    
    // Conv output dimensions
    const int conv_out_depth = (input_depth + 2 * padding_d - kernel_d) / stride_d + 1;
    const int conv_out_height = (input_height + 2 * padding_h - kernel_h) / stride_h + 1;
    const int conv_out_width = (input_width + 2 * padding_w - kernel_w) / stride_w + 1;
    
    // Pool output dimensions
    const int pool_out_depth = (conv_out_depth + 2 * pool_padding_d - pool_kernel_d) / pool_stride_d + 1;
    const int pool_out_height = (conv_out_height + 2 * pool_padding_h - pool_kernel_h) / pool_stride_h + 1;
    const int pool_out_width = (conv_out_width + 2 * pool_padding_w - pool_kernel_w) / pool_stride_w + 1;
    
    const int total_elements = batch_size * out_channels * global_avg_out_depth * global_avg_out_height * global_avg_out_width;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_conv_fused_op_forward", ([&] {
        fused_conv_fused_op_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            conv_weight.data_ptr<scalar_t>(),
            conv_bias.data_ptr<scalar_t>(),
            static_cast<scalar_t>(divisor),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_depth,
            input_height,
            input_width,
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
            padding_d,
            padding_h,
            padding_w,
            conv_out_depth,
            conv_out_height,
            conv_out_width,
            pool_kernel_d,
            pool_kernel_h,
            pool_kernel_w,
            pool_stride_d,
            pool_stride_h,
            pool_stride_w,
            pool_padding_d,
            pool_padding_h,
            pool_padding_w,
            pool_out_depth,
            pool_out_height,
            pool_out_width,
            global_avg_out_depth,
            global_avg_out_height,
            global_avg_out_width,
            sum_dim
        );
    }));
    
    cudaDeviceSynchronize();
}

// Second kernel for sum reduction
void sum_reduction_forward(
    torch::Tensor input,
    torch::Tensor output,
    int reduction_size,
    int output_size
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    const int threads = 256;
    const int blocks = (output_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduction_forward", ([&] {
        sum_reduction_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduction_size,
            output_size
        );
    }));
    
    cudaDeviceSynchronize();
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_fused_op_forward(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    float divisor,
    torch::Tensor bias,
    torch::Tensor output,
    int out_channels,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int pool_kernel_d, int pool_kernel_h, int pool_kernel_w,
    int pool_stride_d, int pool_stride_h, int pool_stride_w,
    int pool_padding_d, int pool_padding_h, int pool_padding_w,
    int global_avg_out_depth, int global_avg_out_height, int global_avg_out_width,
    int sum_dim
);

void sum_reduction_forward(
    torch::Tensor input,
    torch::Tensor output,
    int reduction_size,
    int output_size
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_fused_op", &fused_conv_fused_op_forward, "Fused conv and operations forward");
    m.def("sum_reduction", &sum_reduction_forward, "Sum reduction forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    max_pool_kernel_size,
    max_pool_stride,
    max_pool_padding,
    max_pool_dilation,
    max_pool_ceil_mode,
    max_pool_return_indices,
    global_avg_pool_output_size,
    divisor,
    bias,
    sum_dim,
):
    # Extract dimensions
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    out_channels = conv_weight.shape[0]
    
    # Conv kernel dimensions
    kernel_d, kernel_h, kernel_w = conv_weight.shape[2], conv_weight.shape[3], conv_weight.shape[4]
    
    # Conv stride and padding
    stride_d, stride_h, stride_w = conv_stride
    padding_d, padding_h, padding_w = conv_padding
    
    # Pool parameters
    pool_kernel_d, pool_kernel_h, pool_kernel_w = max_pool_kernel_size
    pool_stride_d, pool_stride_h, pool_stride_w = max_pool_stride
    pool_padding_d, pool_padding_h, pool_padding_w = max_pool_padding
    
    # Global avg pool output size
    global_avg_out_depth, global_avg_out_height, global_avg_out_width = global_avg_pool_output_size
    
    # Create output tensor for fused operations
    fused_output_shape = (batch_size, out_channels, global_avg_out_depth, global_avg_out_height, global_avg_out_width)
    fused_output = torch.empty(fused_output_shape, device=x.device, dtype=x.dtype)
    
    # Call the fused kernel for convolution and all following operations except sum
    fused_ext.fused_conv_fused_op(
        x, conv_weight, conv_bias, divisor, bias, fused_output,
        out_channels,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        pool_kernel_d, pool_kernel_h, pool_kernel_w,
        pool_stride_d, pool_stride_h, pool_stride_w,
        pool_padding_d, pool_padding_h, pool_padding_w,
        global_avg_out_depth, global_avg_out_height, global_avg_out_width,
        sum_dim
    )
    
    # Apply sum reduction
    if sum_dim == 1:  # Sum over channel dimension
        output_shape = (batch_size, global_avg_out_depth, global_avg_out_height, global_avg_out_width)
        final_output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
        reduction_size = out_channels
        output_size = batch_size * global_avg_out_depth * global_avg_out_height * global_avg_out_width
        fused_ext.sum_reduction(fused_output, final_output, reduction_size, output_size)
        return final_output
    else:
        return torch.sum(fused_output, dim=sum_dim)

batch_size   = 128  
in_channels  = 8            
out_channels = 16  
depth = 16; height = width = 64 
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
