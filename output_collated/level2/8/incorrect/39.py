# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_061658/code_5.py
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
from torch.utils.cpp_extension import load_inline

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void fused_conv_div_pool_avgpool_bias_sum_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_d, const int input_h, const int input_w,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int conv_stride_d, const int conv_stride_h, const int conv_stride_w,
    const int conv_pad_d, const int conv_pad_h, const int conv_pad_w,
    const int conv_dilation_d, const int conv_dilation_h, const int conv_dilation_w,
    const int pool_kernel_d, const int pool_kernel_h, const int pool_kernel_w,
    const int pool_stride_d, const int pool_stride_h, const int pool_stride_w,
    const int pool_pad_d, const int pool_pad_h, const int pool_pad_w,
    const int output_d, const int output_h, const int output_w,
    const int pooled_d, const int pooled_h, const int pooled_w,
    const int global_pool_d, const int global_pool_h, const int global_pool_w,
    const float divisor,
    const int sum_dim
) {
    // We parallelize across the batch and output channels for the final reduced output
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * global_pool_d * global_pool_h * global_pool_w;

    if (tid >= total_output_elements) return;

    int b = (tid / (out_channels * global_pool_d * global_pool_h * global_pool_w));
    int oc = (tid / (global_pool_d * global_pool_h * global_pool_w)) % out_channels;
    int gd = (tid / (global_pool_h * global_pool_w)) % global_pool_d;
    int gh = (tid / global_pool_w) % global_pool_h;
    int gw = tid % global_pool_w;

    // Compute the start and end indices of the patch in the pooled tensor that contributes to this global pool output
    float global_pool_sum = 0.0f;
    int pool_d_start = gd * (pooled_d / global_pool_d);
    int pool_h_start = gh * (pooled_h / global_pool_h);
    int pool_w_start = gw * (pooled_w / global_pool_w);
    
    int pool_d_end = (gd + 1) * (pooled_d / global_pool_d);
    int pool_h_end = (gh + 1) * (pooled_h / global_pool_h);
    int pool_w_end = (gw + 1) * (pooled_w / global_pool_w);

    // Iterate through all elements in the pooling window that contribute to this global average pool
    for (int pd = pool_d_start; pd < pool_d_end; ++pd) {
        for (int ph = pool_h_start; ph < pool_h_end; ++ph) {
            for (int pw = pool_w_start; pw < pool_w_end; ++pw) {
                // Now we compute the value at (b, oc, pd, ph, pw) after conv, div, pool
                
                // To compute the convolutional output that led to this pooled element,
                // we need to know the receptive field in the conv output that gets pooled to (pd, ph, pw)
                
                int conv_out_d_start = pd * pool_stride_d - pool_pad_d;
                int conv_out_h_start = ph * pool_stride_h - pool_pad_h;
                int conv_out_w_start = pw * pool_stride_w - pool_pad_w;
                
                float max_val = -1e30f;
                
                // The pooling window in conv output space
                for (int kd = 0; kd < pool_kernel_d; ++kd) {
                    int conv_d = conv_out_d_start + kd;
                    if (conv_d < 0 || conv_d >= output_d) continue;
                    
                    for (int kh = 0; kh < pool_kernel_h; ++kh) {
                        int conv_h = conv_out_h_start + kh;
                        if (conv_h < 0 || conv_h >= output_h) continue;
                        
                        for (int kw = 0; kw < pool_kernel_w; ++kw) {
                            int conv_w = conv_out_w_start + kw;
                            if (conv_w < 0 || conv_w >= output_w) continue;
                            
                            // Compute the convolution result for this position (b, oc, conv_d, conv_h, conv_w)
                            float conv_sum = conv_bias[oc];
                            
                            // Convolve over input channels and kernel
                            for (int ic = 0; ic < in_channels; ++ic) {
                                for (int kd2 = 0; kd2 < kernel_d; ++kd2) {
                                    int in_d = conv_d * conv_stride_d - conv_pad_d + kd2 * conv_dilation_d;
                                    if (in_d < 0 || in_d >= input_d) continue;
                                    
                                    for (int kh2 = 0; kh2 < kernel_h; ++kh2) {
                                        int in_h = conv_h * conv_stride_h - conv_pad_h + kh2 * conv_dilation_h;
                                        if (in_h < 0 || in_h >= input_h) continue;
                                        
                                        for (int kw2 = 0; kw2 < kernel_w; ++kw2) {
                                            int in_w = conv_w * conv_stride_w - conv_pad_w + kw2 * conv_dilation_w;
                                            if (in_w < 0 || in_w >= input_w) continue;
                                            
                                            float in_val = input[(((b * in_channels + ic) * input_d + in_d) * input_h + in_h) * input_w + in_w];
                                            float w_val = weight[(((oc * in_channels + ic) * kernel_d + kd2) * kernel_h + kh2) * kernel_w + kw2];
                                            conv_sum += in_val * w_val;
                                        }
                                    }
                                }
                            }
                            
                            // Apply division and check for max
                            float divided = conv_sum / divisor;
                            if (divided > max_val) {
                                max_val = divided;
                            }
                        }
                    }
                }
                
                // Add bias and accumulate for global average pooling
                global_pool_sum += max_val + bias[oc];
            }
        }
    }
    
    // Normalize by number of elements in the global average pool window
    int elements_per_global_pool = (pooled_d / global_pool_d) * (pooled_h / global_pool_h) * (pooled_w / global_pool_w);
    global_pool_sum /= elements_per_global_pool;
    
    // Store the result
    output[tid] = global_pool_sum;
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_d, int kernel_h, int kernel_w,
    int conv_stride_d, int conv_stride_h, int conv_stride_w,
    int conv_pad_d, int conv_pad_h, int conv_pad_w,
    int conv_dilation_d, int conv_dilation_h, int conv_dilation_w,
    int pool_kernel_d, int pool_kernel_h, int pool_kernel_w,
    int pool_stride_d, int pool_stride_h, int pool_stride_w,
    int pool_pad_d, int pool_pad_h, int pool_pad_w,
    int global_pool_d, int global_pool_h, int global_pool_w,
    float divisor,
    int sum_dim
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_d = input.size(2);
    int input_h = input.size(3);
    int input_w = input.size(4);
    
    int out_channels = weight.size(0);
    
    // Compute output dimensions after conv
    int output_d = (input_d + 2 * conv_pad_d - conv_dilation_d * (kernel_d - 1) - 1) / conv_stride_d + 1;
    int output_h = (input_h + 2 * conv_pad_h - conv_dilation_h * (kernel_h - 1) - 1) / conv_stride_h + 1;
    int output_w = (input_w + 2 * conv_pad_w - conv_dilation_w * (kernel_w - 1) - 1) / conv_stride_w + 1;
    
    // Compute output dimensions after pooling
    int pooled_d = (output_d + 2 * pool_pad_d - pool_kernel_d) / pool_stride_d + 1;
    int pooled_h = (output_h + 2 * pool_pad_h - pool_kernel_h) / pool_stride_h + 1;
    int pooled_w = (output_w + 2 * pool_pad_w - pool_kernel_w) / pool_stride_w + 1;
    
    // Total number of elements in the final output
    int total_output_elements = batch_size * out_channels * global_pool_d * global_pool_h * global_pool_w;
    
    int threads = 256;
    int blocks = (total_output_elements + threads - 1) / threads;
    blocks = min(blocks, 65535); // Cap blocks to avoid launch bounds issues
    
    fused_conv_div_pool_avgpool_bias_sum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        kernel_d, kernel_h, kernel_w,
        conv_stride_d, conv_stride_h, conv_stride_w,
        conv_pad_d, conv_pad_h, conv_pad_w,
        conv_dilation_d, conv_dilation_h, conv_dilation_w,
        pool_kernel_d, pool_kernel_h, pool_kernel_w,
        pool_stride_d, pool_stride_h, pool_stride_w,
        pool_pad_d, pool_pad_h, pool_pad_w,
        output_d, output_h, output_w,
        pooled_d, pooled_h, pooled_w,
        global_pool_d, global_pool_h, global_pool_w,
        divisor,
        sum_dim
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_d, int kernel_h, int kernel_w,
    int conv_stride_d, int conv_stride_h, int conv_stride_w,
    int conv_pad_d, int conv_pad_h, int conv_pad_w,
    int conv_dilation_d, int conv_dilation_h, int conv_dilation_w,
    int pool_kernel_d, int pool_kernel_h, int pool_kernel_w,
    int pool_stride_d, int pool_stride_h, int pool_stride_w,
    int pool_pad_d, int pool_pad_h, int pool_pad_w,
    int global_pool_d, int global_pool_h, int global_pool_w,
    float divisor,
    int sum_dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Convolution + Division + Pooling + Global Avg Pool + Bias + Sum kernel");
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
    # Ensure inputs are contiguous
    x = x.contiguous()
    conv_weight = conv_weight.contiguous()
    conv_bias = conv_bias.contiguous()
    bias = bias.contiguous()
    
    # Extract kernel dimensions
    kernel_d, kernel_h, kernel_w = conv_weight.shape[-3:]
    
    # Extract convolution parameters
    conv_stride_d, conv_stride_h, conv_stride_w = conv_stride
    conv_pad_d, conv_pad_h, conv_pad_w = conv_padding
    conv_dilation_d, conv_dilation_h, conv_dilation_w = conv_dilation
    
    # Extract pooling parameters
    pool_kernel_d, pool_kernel_h, pool_kernel_w = max_pool_kernel_size
    pool_stride_d, pool_stride_h, pool_stride_w = max_pool_stride
    pool_pad_d, pool_pad_h, pool_pad_w = max_pool_padding
    
    # Global average pool output size
    global_pool_d, global_pool_h, global_pool_w = global_avg_pool_output_size
    
    # Create output tensor with appropriate size
    batch_size = x.shape[0]
    out_channels = conv_weight.shape[0]
    output = torch.empty((batch_size, out_channels, global_pool_d, global_pool_h, global_pool_w), 
                         dtype=x.dtype, device=x.device)
    
    # Call the fused kernel
    fused_ext.fused_op(
        x, conv_weight, conv_bias, bias, output,
        kernel_d, kernel_h, kernel_w,
        conv_stride_d, conv_stride_h, conv_stride_w,
        conv_pad_d, conv_pad_h, conv_pad_w,
        conv_dilation_d, conv_dilation_h, conv_dilation_w,
        pool_kernel_d, pool_kernel_h, pool_kernel_w,
        pool_stride_d, pool_stride_h, pool_stride_w,
        pool_pad_d, pool_pad_h, pool_pad_w,
        global_pool_d, global_pool_h, global_pool_w,
        divisor,
        sum_dim
    )
    
    # Apply the final sum reduction along the specified dimension
    output = torch.sum(output, dim=sum_dim)
    
    return output

# Constants as in the original code
batch_size = 128
in_channels = 8
out_channels = 16
depth = height = width = 64
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
