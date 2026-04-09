# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_055948/code_3.py
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

# --- CUDA Kernel ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_1D_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void fused_op_kernel(
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
    const int max_pool_kd, const int max_pool_kh, const int max_pool_kw,
    const int max_pool_stride_d, const int max_pool_stride_h, const int max_pool_stride_w,
    const int max_pool_pad_d, const int max_pool_pad_h, const int max_pool_pad_w,
    const int global_avg_pool_d, const int global_avg_pool_h, const int global_avg_pool_w,
    const float divisor
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    const int out_d = (input_d + 2 * conv_pad_d - conv_dilation_d * (kernel_d - 1) - 1) / conv_stride_d + 1;
    const int out_h = (input_h + 2 * conv_pad_h - conv_dilation_h * (kernel_h - 1) - 1) / conv_stride_h + 1;
    const int out_w = (input_w + 2 * conv_pad_w - conv_dilation_w * (kernel_w - 1) - 1) / conv_stride_w + 1;

    const int pooled_d = (out_d + 2 * max_pool_pad_d - max_pool_kd) / max_pool_stride_d + 1;
    const int pooled_h = (out_h + 2 * max_pool_pad_h - max_pool_kh) / max_pool_stride_h + 1;
    const int pooled_w = (out_w + 2 * max_pool_pad_w - max_pool_kw) / max_pool_stride_w + 1;

    // Loop over output elements assigned to this thread
    for (int idx = tid; idx < batch_size * out_channels; idx += total_threads) {
        const int b = idx / out_channels;
        const int c_out = idx % out_channels;

        float sum_val = 0.0f;

        // Step 1: Conv3D + Divisor
        for (int pd = 0; pd < pooled_d; ++pd) {
            for (int ph = 0; ph < pooled_h; ++ph) {
                for (int pw = 0; pw < pooled_w; ++pw) {
                    float avg_pool_sum = 0.0f;
                    int avg_pool_count = 0;

                    // For each element in the avg pool window
                    for (int i = 0; i < global_avg_pool_d; ++i) {
                        for (int j = 0; j < global_avg_pool_h; ++j) {
                            for (int k = 0; k < global_avg_pool_w; ++k) {
                                // Map avg pool index to max pool index
                                int mpd = pd * global_avg_pool_d + i;
                                int mph = ph * global_avg_pool_h + j;
                                int mpw = pw * global_avg_pool_w + k;

                                float max_val = -1e30f;

                                // For each element in the max pool window
                                for (int md = 0; md < max_pool_kd; ++md) {
                                    for (int mh = 0; mh < max_pool_kh; ++mh) {
                                        for (int mw = 0; mw < max_pool_kw; ++mw) {
                                            // Map max pool index to conv output index
                                            int cd = mpd * max_pool_stride_d - max_pool_pad_d + md;
                                            int ch = mph * max_pool_stride_h - max_pool_pad_h + mh;
                                            int cw = mpw * max_pool_stride_w - max_pool_pad_w + mw;

                                            float conv_val = 0.0f;
                                            if (cd >= 0 && cd < out_d && ch >= 0 && ch < out_h && cw >= 0 && cw < out_w) {
                                                // Conv3D computation
                                                for (int kd = 0; kd < kernel_d; ++kd) {
                                                    for (int kh = 0; kh < kernel_h; ++kh) {
                                                        for (int kw = 0; kw < kernel_w; ++kw) {
                                                            int id = cd * conv_stride_d - conv_pad_d + kd * conv_dilation_d;
                                                            int ih = ch * conv_stride_h - conv_pad_h + kh * conv_dilation_h;
                                                            int iw = cw * conv_stride_w - conv_pad_w + kw * conv_dilation_w;

                                                            if (id >= 0 && id < input_d && ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                                                                for (int c_in = 0; c_in < in_channels; ++c_in) {
                                                                    int input_idx = b * (in_channels * input_d * input_h * input_w) +
                                                                                    c_in * (input_d * input_h * input_w) +
                                                                                    id * (input_h * input_w) +
                                                                                    ih * input_w + iw;
                                                                    int weight_idx = c_out * (in_channels * kernel_d * kernel_h * kernel_w) +
                                                                                     c_in * (kernel_d * kernel_h * kernel_w) +
                                                                                     kd * (kernel_h * kernel_w) +
                                                                                     kh * kernel_w + kw;
                                                                    conv_val += input[input_idx] * weight[weight_idx];
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                                conv_val += conv_bias[c_out]; // Add conv bias
                                                conv_val /= divisor; // Division by divisor
                                            }

                                            max_val = fmaxf(max_val, conv_val);
                                        }
                                    }
                                }
                                avg_pool_sum += max_val;
                                avg_pool_count++;
                            }
                        }
                    }

                    float avg_val = avg_pool_sum / avg_pool_count;
                    sum_val += avg_val;
                }
            }
        }

        sum_val += bias[c_out]; // Add bias
        output[idx] = sum_val;
    }
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor bias,
    torch::Tensor output,
    int conv_stride_d, int conv_stride_h, int conv_stride_w,
    int conv_pad_d, int conv_pad_h, int conv_pad_w,
    int conv_dilation_d, int conv_dilation_h, int conv_dilation_w,
    int max_pool_kd, int max_pool_kh, int max_pool_kw,
    int max_pool_stride_d, int max_pool_stride_h, int max_pool_stride_w,
    int max_pool_pad_d, int max_pool_pad_h, int max_pool_pad_w,
    int global_avg_pool_d, int global_avg_pool_h, int global_avg_pool_w,
    float divisor
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_d = input.size(2);
    const int input_h = input.size(3);
    const int input_w = input.size(4);
    
    const int out_channels = weight.size(0);
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);

    const int threads_per_block = 256;
    const int num_blocks = (batch_size * out_channels + threads_per_block - 1) / threads_per_block;

    fused_op_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_d, input_h, input_w,
        kernel_d, kernel_h, kernel_w,
        conv_stride_d, conv_stride_h, conv_stride_w,
        conv_pad_d, conv_pad_h, conv_pad_w,
        conv_dilation_d, conv_dilation_h, conv_dilation_w,
        max_pool_kd, max_pool_kh, max_pool_kw,
        max_pool_stride_d, max_pool_stride_h, max_pool_stride_w,
        max_pool_pad_d, max_pool_pad_h, max_pool_pad_w,
        global_avg_pool_d, global_avg_pool_h, global_avg_pool_w,
        divisor
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
    int conv_stride_d, int conv_stride_h, int conv_stride_w,
    int conv_pad_d, int conv_pad_h, int conv_pad_w,
    int conv_dilation_d, int conv_dilation_h, int conv_dilation_w,
    int max_pool_kd, int max_pool_kh, int max_pool_kw,
    int max_pool_stride_d, int max_pool_stride_h, int max_pool_stride_w,
    int max_pool_pad_d, int max_pool_pad_h, int max_pool_pad_w,
    int global_avg_pool_d, int global_avg_pool_h, int global_avg_pool_w,
    float divisor
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused 3D Conv + Pool + Bias + Sum");
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
    batch_size = x.shape[0]
    out_channels = conv_weight.shape[0]
    
    # Prepare output tensor
    output = torch.zeros((batch_size, out_channels), device=x.device, dtype=x.dtype)
    
    # Unpack parameters
    conv_stride_d, conv_stride_h, conv_stride_w = conv_stride
    conv_pad_d, conv_pad_h, conv_pad_w = conv_padding
    conv_dilation_d, conv_dilation_h, conv_dilation_w = conv_dilation
    
    max_pool_kd, max_pool_kh, max_pool_kw = max_pool_kernel_size
    max_pool_stride_d, max_pool_stride_h, max_pool_stride_w = max_pool_stride
    max_pool_pad_d, max_pool_pad_h, max_pool_pad_w = max_pool_padding
    
    global_avg_pool_d, global_avg_pool_h, global_avg_pool_w = global_avg_pool_output_size
    
    # Call the optimized custom kernel
    fused_ext.fused_op(
        x, conv_weight, conv_bias, bias, output,
        conv_stride_d, conv_stride_h, conv_stride_w,
        conv_pad_d, conv_pad_h, conv_pad_w,
        conv_dilation_d, conv_dilation_h, conv_dilation_w,
        max_pool_kd, max_pool_kh, max_pool_kw,
        max_pool_stride_d, max_pool_stride_h, max_pool_stride_w,
        max_pool_pad_d, max_pool_pad_h, max_pool_pad_w,
        global_avg_pool_d, global_avg_pool_h, global_avg_pool_w,
        divisor
    )
    
    return output

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
