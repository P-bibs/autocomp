# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_055208/code_3.py
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

# Optimized CUDA kernel code
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CeilDiv(a, b) (((a) + (b) - 1) / (b))

// Fused kernel implementation
__global__ void fused_model_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    float* __restrict__ output,
    float divisor,
    const float* __restrict__ bias,
    int batch,
    int in_channels,
    int out_channels,
    int id,
    int ih,
    int iw,
    int od,
    int oh,
    int ow,
    int kd,
    int kh,
    int kw,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int pool_kd,
    int pool_kh,
    int pool_kw,
    int pool_stride_d,
    int pool_stride_h,
    int pool_stride_w,
    int pool_pad_d,
    int pool_pad_h,
    int pool_pad_w,
    int global_out_d,
    int global_out_h,
    int global_out_w,
    int sum_dim
) {
    // Calculate output indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * out_channels * global_out_d * global_out_h * global_out_w;
    
    if (idx >= total_elements) return;
    
    int w = idx % global_out_w;
    int h = (idx / global_out_w) % global_out_h;
    int d = (idx / (global_out_w * global_out_h)) % global_out_d;
    int c = (idx / (global_out_w * global_out_h * global_out_d)) % out_channels;
    int b = idx / (global_out_w * global_out_h * global_out_d * out_channels);
    
    // Map adaptive average pooling indices back to regular pooling output
    int pool_out_d = CeilDiv(od + 2 * pool_pad_d - (pool_kd - 1) - 1, pool_stride_d) + 1;
    int pool_out_h = CeilDiv(oh + 2 * pool_pad_h - (pool_kh - 1) - 1, pool_stride_h) + 1;
    int pool_out_w = CeilDiv(ow + 2 * pool_pad_w - (pool_kw - 1) - 1, pool_stride_w) + 1;
    
    float pooled_value = -1e30f; // Initialize to very small value for max pooling
    
    // Calculate the receptive field in the pooled output space
    int start_d = d * (pool_out_d / global_out_d);
    int end_d = (d + 1) * (pool_out_d / global_out_d);
    int start_h = h * (pool_out_h / global_out_h);
    int end_h = (h + 1) * (pool_out_h / global_out_h);
    int start_w = w * (pool_out_w / global_out_w);
    int end_w = (w + 1) * (pool_out_w / global_out_w);
    
    // Boundary checks
    start_d = max(0, start_d);
    start_h = max(0, start_h);
    start_w = max(0, start_w);
    end_d = min(pool_out_d, end_d);
    end_h = min(pool_out_h, end_h);
    end_w = min(pool_out_w, end_w);
    
    // Iterate through the receptive field in pooled output space
    for (int pd = start_d; pd < end_d; pd++) {
        for (int ph = start_h; ph < end_h; ph++) {
            for (int pw = start_w; pw < end_w; pw++) {
                // Max pooling operation
                float max_val = -1e30f;
                
                // Perform convolution and pooling for this position
                int conv_d_start = pd * pool_stride_d - pool_pad_d;
                int conv_h_start = ph * pool_stride_h - pool_pad_h;
                int conv_w_start = pw * pool_stride_w - pool_pad_w;
                
                // Calculate the convolution output for this pooling window
                for (int kd_idx = 0; kd_idx < pool_kd; kd_idx++) {
                    for (int kh_idx = 0; kh_idx < pool_kh; kh_idx++) {
                        for (int kw_idx = 0; kw_idx < pool_kw; kw_idx++) {
                            int conv_d = conv_d_start + kd_idx;
                            int conv_h = conv_h_start + kh_idx;
                            int conv_w = conv_w_start + kw_idx;
                            
                            // Convolution calculation
                            float conv_sum = 0.0f;
                            if (conv_d >= 0 && conv_d < od && 
                                conv_h >= 0 && conv_h < oh && 
                                conv_w >= 0 && conv_w < ow) {
                                
                                for (int ic = 0; ic < in_channels; ic++) {
                                    for (int kdi = 0; kdi < kd; kdi++) {
                                        for (int khi = 0; khi < kh; khi++) {
                                            for (int kwi = 0; kwi < kw; kwi++) {
                                                int in_d = conv_d * stride_d + kdi * dilation_d - pad_d;
                                                int in_h = conv_h * stride_h + khi * dilation_h - pad_h;
                                                int in_w = conv_w * stride_w + kwi * dilation_w - pad_w;
                                                
                                                if (in_d >= 0 && in_d < id && 
                                                    in_h >= 0 && in_h < ih && 
                                                    in_w >= 0 && in_w < iw) {
                                                    
                                                    int input_idx = b * in_channels * id * ih * iw +
                                                                    ic * id * ih * iw +
                                                                    in_d * ih * iw +
                                                                    in_h * iw +
                                                                    in_w;
                                                                        
                                                    int weight_idx = c * in_channels * kd * kh * kw +
                                                                     ic * kd * kh * kw +
                                                                     kdi * kh * kw +
                                                                     khi * kw +
                                                                     kwi;
                                                                        
                                                    conv_sum += input[input_idx] * weight[weight_idx];
                                                }
                                            }
                                        }
                                    }
                                }
                                
                                conv_sum += conv_bias[c];  // Add bias
                                conv_sum /= divisor;       // Apply divisor
                                max_val = fmaxf(max_val, conv_sum);
                            }
                        }
                    }
                }
                
                pooled_value = fmaxf(pooled_value, max_val);
            }
        }
    }
    
    // Add bias and perform final summation
    if (sum_dim == 1) {
        float final_result = pooled_value + bias[c];
        output[idx] = final_result;
    } else {
        // For other sum dimensions, we'd handle differently
        output[idx] = pooled_value + bias[c];
    }
}

void fused_model_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor bias,
    torch::Tensor output,
    float divisor,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int pool_kd, int pool_kh, int pool_kw,
    int pool_stride_d, int pool_stride_h, int pool_stride_w,
    int pool_pad_d, int pool_pad_h, int pool_pad_w,
    int global_out_d, int global_out_h, int global_out_w,
    int sum_dim
) {
    int batch = input.size(0);
    int in_channels = input.size(1);
    int id = input.size(2);
    int ih = input.size(3);
    int iw = input.size(4);
    
    int out_channels = weight.size(0);
    int kd = weight.size(2);
    int kh = weight.size(3);
    int kw = weight.size(4);
    
    // Calculate output dimensions after conv
    int od = ((id + 2 * pad_d - dilation_d * (kd - 1) - 1) / stride_d) + 1;
    int oh = ((ih + 2 * pad_h - dilation_h * (kh - 1) - 1) / stride_h) + 1;
    int ow = ((iw + 2 * pad_w - dilation_w * (kw - 1) - 1) / stride_w) + 1;
    
    int total_elements = batch * out_channels * global_out_d * global_out_h * global_out_w;
    int threads = 256;
    int blocks = CeilDiv(total_elements, threads);
    
    fused_model_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        divisor,
        bias.data_ptr<float>(),
        batch,
        in_channels,
        out_channels,
        id, ih, iw,
        od, oh, ow,
        kd, kh, kw,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w,
        pool_kd, pool_kh, pool_kw,
        pool_stride_d, pool_stride_h, pool_stride_w,
        pool_pad_d, pool_pad_h, pool_pad_w,
        global_out_d, global_out_h, global_out_w,
        sum_dim
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_model_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor bias,
    torch::Tensor output,
    float divisor,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int pool_kd, int pool_kh, int pool_kw,
    int pool_stride_d, int pool_stride_h, int pool_stride_w,
    int pool_pad_d, int pool_pad_h, int pool_pad_w,
    int global_out_d, int global_out_h, int global_out_w,
    int sum_dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_model_forward", &fused_model_forward, "Fused 3D Conv, Pool, and Reduce kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_model_ext',
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
    # Extract parameters
    stride_d, stride_h, stride_w = conv_stride
    pad_d, pad_h, pad_w = conv_padding
    dilation_d, dilation_h, dilation_w = conv_dilation
    
    pool_kd, pool_kh, pool_kw = max_pool_kernel_size
    pool_stride_d, pool_stride_h, pool_stride_w = max_pool_stride
    pool_pad_d, pool_pad_h, pool_pad_w = max_pool_padding
    
    global_out_d, global_out_h, global_out_w = global_avg_pool_output_size
    
    # Create output tensor
    batch_size = x.size(0)
    out_channels = conv_weight.size(0)
    output = torch.empty((batch_size, out_channels, global_out_d, global_out_h, global_out_w), 
                         dtype=x.dtype, device=x.device)
    
    # Call the fused kernel
    fused_ext.fused_model_forward(
        x, conv_weight, conv_bias, bias, output,
        divisor,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w,
        pool_kd, pool_kh, pool_kw,
        pool_stride_d, pool_stride_h, pool_stride_w,
        pool_pad_d, pool_pad_h, pool_pad_w,
        global_out_d, global_out_h, global_out_w,
        sum_dim
    )
    
    # Perform final summation along the specified dimension
    output = torch.sum(output, dim=sum_dim)
    
    return output

# Standard boilerplate for input generation
batch_size = 128
in_channels = 8
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
