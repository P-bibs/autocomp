# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_061658/code_8.py
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

# CUDA kernel for fused operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ void fused_conv_pool_reduce_kernel(
    const float* __restrict__ input,      // (B, C_in, D, H, W)
    const float* __restrict__ weight,     // (C_out, C_in, K_d, K_h, K_w)
    const float* __restrict__ conv_bias,  // (C_out,)
    const float* __restrict__ bias,       // (C_out, 1, 1, 1)
    float* __restrict__ output,           // (B,)
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d,
    int input_h,
    int input_w,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int conv_stride_d,
    int conv_stride_h,
    int conv_stride_w,
    int conv_pad_d,
    int conv_pad_h,
    int conv_pad_w,
    int conv_dilation_d,
    int conv_dilation_h,
    int conv_dilation_w,
    float divisor,
    int pool_kernel_d,
    int pool_kernel_h,
    int pool_kernel_w,
    int pool_stride_d,
    int pool_stride_h,
    int pool_stride_w,
    int pool_pad_d,
    int pool_pad_h,
    int pool_pad_w,
    int pool_dilation_d,
    int pool_dilation_h,
    int pool_dilation_w,
    int adaptive_d,
    int adaptive_h,
    int adaptive_w
) {
    // Each block handles one output element
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y;
    
    if (batch_idx >= batch_size || out_ch >= out_channels) return;
    
    // Step 1: Conv3D
    // Compute output dimensions after conv
    int conv_out_d = (input_d + 2 * conv_pad_d - conv_dilation_d * (kernel_d - 1) - 1) / conv_stride_d + 1;
    int conv_out_h = (input_h + 2 * conv_pad_h - conv_dilation_h * (kernel_h - 1) - 1) / conv_stride_h + 1;
    int conv_out_w = (input_w + 2 * conv_pad_w - conv_dilation_w * (kernel_w - 1) - 1) / conv_stride_w + 1;
    
    // Shared memory to store conv output for this channel
    extern __shared__ float conv_output[];
    
    // Thread indices for conv output
    int tid = threadIdx.x;
    int total_conv_elements = conv_out_d * conv_out_h * conv_out_w;
    
    // Initialize shared memory
    for (int i = tid; i < total_conv_elements; i += blockDim.x) {
        conv_output[i] = 0.0f;
    }
    __syncthreads();
    
    // Compute convolution for this output channel
    for (int od = 0; od < conv_out_d; od++) {
        for (int oh = 0; oh < conv_out_h; oh++) {
            for (int ow = 0; ow < conv_out_w; ow++) {
                float sum = 0.0f;
                
                for (int kd = 0; kd < kernel_d; kd++) {
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            for (int ic = 0; ic < in_channels; ic++) {
                                int in_d = od * conv_stride_d - conv_pad_d + kd * conv_dilation_d;
                                int in_h = oh * conv_stride_h - conv_pad_h + kh * conv_dilation_h;
                                int in_w = ow * conv_stride_w - conv_pad_w + kw * conv_dilation_w;
                                
                                if (in_d >= 0 && in_d < input_d && 
                                    in_h >= 0 && in_h < input_h && 
                                    in_w >= 0 && in_w < input_w) {
                                    
                                    int input_idx = batch_idx * (in_channels * input_d * input_h * input_w) +
                                                   ic * (input_d * input_h * input_w) +
                                                   in_d * (input_h * input_w) +
                                                   in_h * input_w +
                                                   in_w;
                                                   
                                    int weight_idx = out_ch * (in_channels * kernel_d * kernel_h * kernel_w) +
                                                    ic * (kernel_d * kernel_h * kernel_w) +
                                                    kd * (kernel_h * kernel_w) +
                                                    kh * kernel_w +
                                                    kw;
                                                    
                                    sum += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                }
                
                // Add bias
                sum += conv_bias[out_ch];
                
                // Store in shared memory
                int conv_idx = od * (conv_out_h * conv_out_w) + oh * conv_out_w + ow;
                conv_output[conv_idx] = sum;
            }
        }
    }
    
    __syncthreads();
    
    // Step 2: Max pooling + Adaptive avg pooling + Bias addition
    // Compute output dimensions after max pooling
    int pool_out_d = (conv_out_d + 2 * pool_pad_d - pool_dilation_d * (pool_kernel_d - 1) - 1) / pool_stride_d + 1;
    int pool_out_h = (conv_out_h + 2 * pool_pad_h - pool_dilation_h * (pool_kernel_h - 1) - 1) / pool_stride_h + 1;
    int pool_out_w = (conv_out_w + 2 * pool_pad_w - pool_dilation_w * (pool_kernel_w - 1) - 1) / pool_stride_w + 1;
    
    float channel_sum = 0.0f;
    
    // For adaptive pooling, we need to map each output position to input regions
    for (int pod = 0; pod < adaptive_d; pod++) {
        int start_d = (pod * pool_out_d) / adaptive_d;
        int end_d = ((pod + 1) * pool_out_d + adaptive_d - 1) / adaptive_d;
        
        for (int poh = 0; poh < adaptive_h; poh++) {
            int start_h = (poh * pool_out_h) / adaptive_h;
            int end_h = ((poh + 1) * pool_out_h + adaptive_h - 1) / adaptive_h;
            
            for (int pow = 0; pow < adaptive_w; pow++) {
                int start_w = (pow * pool_out_w) / adaptive_w;
                int end_w = ((pow + 1) * pool_out_w + adaptive_w - 1) / adaptive_w;
                
                float sum_val = 0.0f;
                int count = 0;
                
                // Perform max pooling within this adaptive region
                for (int pd = start_d; pd < end_d; pd++) {
                    for (int ph = start_h; ph < end_h; ph++) {
                        for (int pw = start_w; pw < end_w; pw++) {
                            // Max pooling operation
                            float max_val = -FLT_MAX;
                            
                            for (int kd = 0; kd < pool_kernel_d; kd++) {
                                for (int kh = 0; kh < pool_kernel_h; kh++) {
                                    for (int kw = 0; kw < pool_kernel_w; kw++) {
                                        int in_d = pd * pool_stride_d - pool_pad_d + kd * pool_dilation_d;
                                        int in_h = ph * pool_stride_h - pool_pad_h + kh * pool_dilation_h;
                                        int in_w = pw * pool_stride_w - pool_pad_w + kw * pool_dilation_w;
                                        
                                        if (in_d >= 0 && in_d < conv_out_d && 
                                            in_h >= 0 && in_h < conv_out_h && 
                                            in_w >= 0 && in_w < conv_out_w) {
                                            
                                            int conv_idx = in_d * (conv_out_h * conv_out_w) + 
                                                          in_h * conv_out_w + 
                                                          in_w;
                                            float val = conv_output[conv_idx] / divisor;
                                            max_val = fmaxf(max_val, val);
                                        }
                                    }
                                }
                            }
                            
                            if (max_val > -FLT_MAX) {
                                sum_val += max_val;
                                count++;
                            }
                        }
                    }
                }
                
                // Average within the adaptive pooling region
                float avg_val = (count > 0) ? (sum_val / count) : 0.0f;
                
                // Add bias and accumulate
                channel_sum += avg_val + bias[out_ch];
            }
        }
    }
    
    // Atomic add to final result
    atomicAdd(&output[batch_idx], channel_sum);
}

void fused_model_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_d, int kernel_h, int kernel_w,
    int conv_stride_d, int conv_stride_h, int conv_stride_w,
    int conv_pad_d, int conv_pad_h, int conv_pad_w,
    int conv_dilation_d, int conv_dilation_h, int conv_dilation_w,
    float divisor,
    int pool_kernel_d, int pool_kernel_h, int pool_kernel_w,
    int pool_stride_d, int pool_stride_h, int pool_stride_w,
    int pool_pad_d, int pool_pad_h, int pool_pad_w,
    int pool_dilation_d, int pool_dilation_h, int pool_dilation_w,
    int adaptive_d, int adaptive_h, int adaptive_w
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_d = input.size(2);
    auto input_h = input.size(3);
    auto input_w = input.size(4);
    
    auto out_channels = weight.size(0);
    
    // Compute conv output dimensions
    int conv_out_d = (input_d + 2 * conv_pad_d - conv_dilation_d * (kernel_d - 1) - 1) / conv_stride_d + 1;
    int conv_out_h = (input_h + 2 * conv_pad_h - conv_dilation_h * (kernel_h - 1) - 1) / conv_stride_h + 1;
    int conv_out_w = (input_w + 2 * conv_pad_w - conv_dilation_w * (kernel_w - 1) - 1) / conv_stride_w + 1;
    
    int total_conv_elements = conv_out_d * conv_out_h * conv_out_w;
    
    dim3 blocks(batch_size, out_channels);
    dim3 threads(min(1024, total_conv_elements));
    
    size_t shared_mem_size = total_conv_elements * sizeof(float);
    
    fused_conv_pool_reduce_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d,
        input_h,
        input_w,
        kernel_d,
        kernel_h,
        kernel_w,
        conv_stride_d,
        conv_stride_h,
        conv_stride_w,
        conv_pad_d,
        conv_pad_h,
        conv_pad_w,
        conv_dilation_d,
        conv_dilation_h,
        conv_dilation_w,
        divisor,
        pool_kernel_d,
        pool_kernel_h,
        pool_kernel_w,
        pool_stride_d,
        pool_stride_h,
        pool_stride_w,
        pool_pad_d,
        pool_pad_h,
        pool_pad_w,
        pool_dilation_d,
        pool_dilation_h,
        pool_dilation_w,
        adaptive_d,
        adaptive_h,
        adaptive_w
    );
    
    cudaDeviceSynchronize();
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_model_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_d, int kernel_h, int kernel_w,
    int conv_stride_d, int conv_stride_h, int conv_stride_w,
    int conv_pad_d, int conv_pad_h, int conv_pad_w,
    int conv_dilation_d, int conv_dilation_h, int conv_dilation_w,
    float divisor,
    int pool_kernel_d, int pool_kernel_h, int pool_kernel_w,
    int pool_stride_d, int pool_stride_h, int pool_stride_w,
    int pool_pad_d, int pool_pad_h, int pool_pad_w,
    int pool_dilation_d, int pool_dilation_h, int pool_dilation_w,
    int adaptive_d, int adaptive_h, int adaptive_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_model_forward", &fused_model_forward, "Fused model operations kernel");
}
"""

fused_ext = load_inline(
    name='fused_model',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

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
    # Create output tensor
    result = torch.zeros(batch_size, dtype=x.dtype, device=x.device)
    
    # Call fused kernel that performs:
    # 1. Conv3d
    # 2. Division by divisor
    # 3. Max pooling
    # 4. Adaptive average pooling
    # 5. Bias addition
    # 6. Sum reduction over channels
    fused_ext.fused_model_forward(
        x,
        conv_weight,
        conv_bias,
        bias,
        result,
        kernel_size[0], kernel_size[1], kernel_size[2],
        conv_stride[0], conv_stride[1], conv_stride[2],
        conv_padding[0], conv_padding[1], conv_padding[2],
        conv_dilation[0], conv_dilation[1], conv_dilation[2],
        divisor,
        max_pool_kernel_size[0], max_pool_kernel_size[1], max_pool_kernel_size[2],
        max_pool_stride[0], max_pool_stride[1], max_pool_stride[2],
        max_pool_padding[0], max_pool_padding[1], max_pool_padding[2],
        max_pool_dilation[0], max_pool_dilation[1], max_pool_dilation[2],
        global_avg_pool_output_size[0], global_avg_pool_output_size[1], global_avg_pool_output_size[2]
    )
    
    return result
