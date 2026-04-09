# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_053523/code_1.py
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

# CUDA kernel for fused 3D convolution with post-processing
# This kernel fuses:
# 1. Conv3D (tiled implementation)
# 2. Division by divisor
# 3. Bias addition
# 4. Global average pooling (adaptive)
# 5. Channel-wise summation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_op_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias_tensor,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int depth,
    const int height,
    const int width,
    const int out_channels,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int dilation_d,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int pool_kd,
    const int pool_kh,
    const int pool_kw,
    const int pool_stride_d,
    const int pool_stride_h,
    const int pool_stride_w,
    const int pool_padding_d,
    const int pool_padding_h,
    const int pool_padding_w,
    const int pool_dilation_d,
    const int pool_dilation_h,
    const int pool_dilation_w,
    const bool ceil_mode,
    const float divisor
) {
    const int oc = blockIdx.x;
    const int b = blockIdx.y;
    
    if (oc >= out_channels || b >= batch_size) return;
    
    const int g = oc / (out_channels / groups);
    const int ic_start = g * (in_channels / groups);
    const int ic_end = (g + 1) * (in_channels / groups);

    // Shared memory to accumulate intermediate results
    extern __shared__ float sdata[];
    
    // Thread indices
    const int tid = threadIdx.x;
    const int threads_per_block = blockDim.x;
    
    // Calculate conv output dimensions
    const int out_d = (depth + 2 * padding_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    const int out_h = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_w = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    // Calculate max pool output dimensions
    const int pooled_d = ceil_mode ? 
        (int)ceilf((float)(out_d + 2 * pool_padding_d - pool_dilation_d * (pool_kd - 1) - 1) / pool_stride_d + 1) :
        (out_d + 2 * pool_padding_d - pool_dilation_d * (pool_kd - 1) - 1) / pool_stride_d + 1;
    const int pooled_h = ceil_mode ? 
        (int)ceilf((float)(out_h + 2 * pool_padding_h - pool_dilation_h * (pool_kh - 1) - 1) / pool_stride_h + 1) :
        (out_h + 2 * pool_padding_h - pool_dilation_h * (pool_kh - 1) - 1) / pool_stride_h + 1;
    const int pooled_w = ceil_mode ? 
        (int)ceilf((float)(out_w + 2 * pool_padding_w - pool_dilation_w * (pool_kw - 1) - 1) / pool_stride_w + 1) :
        (out_w + 2 * pool_padding_w - pool_dilation_w * (pool_kw - 1) - 1) / pool_stride_w + 1;
    
    // Clamp to ensure we don't exceed bounds when ceil_mode is true
    const int effective_pooled_d = min(pooled_d, 
        (out_d + 2 * pool_padding_d - pool_dilation_d * (pool_kd - 1) - 1 + pool_stride_d - 1) / pool_stride_d + 1);
    const int effective_pooled_h = min(pooled_h, 
        (out_h + 2 * pool_padding_h - pool_dilation_h * (pool_kh - 1) - 1 + pool_stride_h - 1) / pool_stride_h + 1);
    const int effective_pooled_w = min(pooled_w, 
        (out_w + 2 * pool_padding_w - pool_dilation_w * (pool_kw - 1) - 1 + pool_stride_w - 1) / pool_stride_w + 1);
    
    float sum_val = 0.0f;
    
    // Process each spatial location in the final pooled output
    for (int pd = 0; pd < effective_pooled_d; pd++) {
        for (int ph = 0; ph < effective_pooled_h; ph++) {
            for (int pw = 0; pw < effective_pooled_w; pw++) {
                float pooled_val = -FLT_MAX; // For max pooling
                
                // Check all positions in the pooling window
                for (int kd = 0; kd < pool_kd; kd++) {
                    for (int kh = 0; kh < pool_kh; kh++) {
                        for (int kw = 0; kw < pool_kw; kw++) {
                            int d = pd * pool_stride_d - pool_padding_d + kd * pool_dilation_d;
                            int h = ph * pool_stride_h - pool_padding_h + kh * pool_dilation_h;
                            int w = pw * pool_stride_w - pool_padding_w + kw * pool_dilation_w;
                            
                            if (d >= 0 && d < out_d && h >= 0 && h < out_h && w >= 0 && w < out_w) {
                                // Compute convolution at this position
                                float conv_val = 0.0f;
                                
                                // Convolution loop
                                for (int ic = ic_start + tid; ic < ic_end; ic += threads_per_block) {
                                    for (int kd_conv = 0; kd_conv < kernel_d; kd_conv++) {
                                        for (int kh_conv = 0; kh_conv < kernel_h; kh_conv++) {
                                            for (int kw_conv = 0; kw_conv < kernel_w; kw_conv++) {
                                                int id = d * stride_d - padding_d + kd_conv * dilation_d;
                                                int ih = h * stride_h - padding_h + kh_conv * dilation_h;
                                                int iw = w * stride_w - padding_w + kw_conv * dilation_w;
                                                
                                                if (id >= 0 && id < depth && ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                                    int input_idx = b * (in_channels * depth * height * width) +
                                                                    ic * (depth * height * width) +
                                                                    id * (height * width) +
                                                                    ih * width +
                                                                    iw;
                                                    int weight_idx = oc * (in_channels * kernel_d * kernel_h * kernel_w) +
                                                                     ic * (kernel_d * kernel_h * kernel_w) +
                                                                     kd_conv * (kernel_h * kernel_w) +
                                                                     kh_conv * kernel_w +
                                                                     kw_conv;
                                                    conv_val += input[input_idx] * weight[weight_idx];
                                                }
                                            }
                                        }
                                    }
                                }
                                
                                // Reduction across threads in block
                                sdata[tid] = conv_val;
                                __syncthreads();
                                
                                // Reduction in shared memory
                                for (int s = threads_per_block / 2; s > 0; s >>= 1) {
                                    if (tid < s) {
                                        sdata[tid] += sdata[tid + s];
                                    }
                                    __syncthreads();
                                }
                                
                                float final_conv_val = sdata[0] / divisor + bias_tensor[oc];
                                
                                if (tid == 0 && final_conv_val > pooled_val) {
                                    pooled_val = final_conv_val;
                                }
                                
                                __syncthreads(); // Reset for next iteration
                            }
                        }
                    }
                }
                
                // Only one thread adds to the sum (after max pooling)
                if (tid == 0) {
                    sum_val += pooled_val;
                }
            }
        }
    }
    
    // Since we're doing adaptive average pool to size (1,1,1), divide by total elements
    if (tid == 0) {
        float avg_pooled = sum_val / (effective_pooled_d * effective_pooled_h * effective_pooled_w);
        output[b * out_channels + oc] = avg_pooled;
    }
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_tensor,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups,
    int pool_kd, int pool_kh, int pool_kw,
    int pool_stride_d, int pool_stride_h, int pool_stride_w,
    int pool_padding_d, int pool_padding_h, int pool_padding_w,
    int pool_dilation_d, int pool_dilation_h, int pool_dilation_w,
    bool ceil_mode,
    float divisor
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int depth = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);
    const int out_channels = weight.size(0);
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);
    
    // Grid and block dimensions
    const dim3 blocks(out_channels, batch_size);
    const dim3 threads(32); // Use 32 threads per block for reduction
    
    // Shared memory size
    const int shared_mem_size = threads.x * sizeof(float);
    
    fused_op_forward_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_tensor.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        depth,
        height,
        width,
        out_channels,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w,
        groups,
        pool_kd, pool_kh, pool_kw,
        pool_stride_d, pool_stride_h, pool_stride_w,
        pool_padding_d, pool_padding_h, pool_padding_w,
        pool_dilation_d, pool_dilation_h, pool_dilation_w,
        ceil_mode,
        divisor
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_tensor,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups,
    int pool_kd, int pool_kh, int pool_kw,
    int pool_stride_d, int pool_stride_h, int pool_stride_w,
    int pool_padding_d, int pool_padding_h, int pool_padding_w,
    int pool_dilation_d, int pool_dilation_h, int pool_dilation_w,
    bool ceil_mode,
    float divisor
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused 3D Conv + Post-processing operation");
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
    # Ensure inputs are on CUDA
    if not x.is_cuda:
        x = x.cuda()
    if not conv_weight.is_cuda:
        conv_weight = conv_weight.cuda()
    if not bias.is_cuda:
        bias = bias.cuda()
        
    # Prepare output tensor
    batch_size = x.size(0)
    out_channels = conv_weight.size(0)
    output = torch.empty((batch_size, out_channels), dtype=x.dtype, device=x.device)
    
    # Handle tuple parameters
    if isinstance(conv_stride, int):
        conv_stride = (conv_stride, conv_stride, conv_stride)
    if isinstance(conv_padding, int):
        conv_padding = (conv_padding, conv_padding, conv_padding)
    if isinstance(conv_dilation, int):
        conv_dilation = (conv_dilation, conv_dilation, conv_dilation)
        
    if isinstance(max_pool_kernel_size, int):
        max_pool_kernel_size = (max_pool_kernel_size, max_pool_kernel_size, max_pool_kernel_size)
    if isinstance(max_pool_stride, int):
        max_pool_stride = (max_pool_stride, max_pool_stride, max_pool_stride)
    if isinstance(max_pool_padding, int):
        max_pool_padding = (max_pool_padding, max_pool_padding, max_pool_padding)
    if isinstance(max_pool_dilation, int):
        max_pool_dilation = (max_pool_dilation, max_pool_dilation, max_pool_dilation)

    # Call the fused operation
    fused_ext.fused_op(
        x, conv_weight, bias, output,
        conv_stride[0], conv_stride[1], conv_stride[2],
        conv_padding[0], conv_padding[1], conv_padding[2],
        conv_dilation[0], conv_dilation[1], conv_dilation[2],
        conv_groups,
        max_pool_kernel_size[0], max_pool_kernel_size[1], max_pool_kernel_size[2],
        max_pool_stride[0], max_pool_stride[1], max_pool_stride[2],
        max_pool_padding[0], max_pool_padding[1], max_pool_padding[2],
        max_pool_dilation[0], max_pool_dilation[1], max_pool_dilation[2],
        max_pool_ceil_mode,
        divisor
    )
    
    # Sum over the specified dimension (channel dim in this case is 1)
    return torch.sum(output, dim=sum_dim)

# Constants
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
