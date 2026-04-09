# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_054338/code_1.py
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

# --- CUDA Code ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_MAX_THREADS_PER_BLOCK 1024

__device__ inline int index_to_position(int n, int c, int d, int h, int w,
                                        int C, int D, int H, int W) {
    return ((n * C + c) * D + d) * H * W + h * W + w;
}

__device__ inline bool within_bounds(int d, int h, int w, int D, int H, int W) {
    return (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W);
}

__global__ void fused_conv_pool_poolavg_add_sum_kernel(
    const float* __restrict__ input,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias_data,
    float* __restrict__ output,
    const int batch_size, const int in_channels, const int out_channels,
    const int in_depth, const int in_height, const int in_width,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int conv_stride_d, const int conv_stride_h, const int conv_stride_w,
    const int conv_pad_d, const int conv_pad_h, const int conv_pad_w,
    const int pool_kernel_d, const int pool_kernel_h, const int pool_kernel_w,
    const int pool_stride_d, const int pool_stride_h, const int pool_stride_w,
    const int pool_pad_d, const int pool_pad_h, const int pool_pad_w,
    const int out_depth, const int out_height, const int out_width,
    const int pool_out_depth, const int pool_out_height, const int pool_out_width,
    const int adaptive_out_depth, const int adaptive_out_height, const int adaptive_out_width,
    const float divisor,
    const int sum_dim
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = batch_size * out_channels * adaptive_out_depth * adaptive_out_height * adaptive_out_width;
    
    if (tid >= total_threads) return;

    const int out_w_idx = tid % adaptive_out_width;
    const int out_h_idx = (tid / adaptive_out_width) % adaptive_out_height;
    const int out_d_idx = (tid / (adaptive_out_width * adaptive_out_height)) % adaptive_out_depth;
    const int out_c_idx = (tid / (adaptive_out_width * adaptive_out_height * adaptive_out_depth)) % out_channels;
    const int out_n_idx = tid / (adaptive_out_width * adaptive_out_height * adaptive_out_depth * out_channels);

    const float d_scale = static_cast<float>(pool_out_depth) / static_cast<float>(adaptive_out_depth);
    const float h_scale = static_cast<float>(pool_out_height) / static_cast<float>(adaptive_out_height);
    const float w_scale = static_cast<float>(pool_out_width) / static_cast<float>(adaptive_out_width);

    const int start_d = static_cast<int>(floorf(out_d_idx * d_scale));
    const int end_d = static_cast<int>(ceilf((out_d_idx + 1) * d_scale));
    const int start_h = static_cast<int>(floorf(out_h_idx * h_scale));
    const int end_h = static_cast<int>(ceilf((out_h_idx + 1) * h_scale));
    const int start_w = static_cast<int>(floorf(out_w_idx * w_scale));
    const int end_w = static_cast<int>(ceilf((out_w_idx + 1) * w_scale));

    float sum = 0.0f;
    int count = 0;

    for (int pd = start_d; pd < end_d; pd++) {
        for (int ph = start_h; ph < end_h; ph++) {
            for (int pw = start_w; pw < end_w; pw++) {
                if (pd < 0 || pd >= pool_out_depth || ph < 0 || ph >= pool_out_height || pw < 0 || pw >= pool_out_width) continue;
                
                // Max pool to conv output mapping
                const int start_md = pd * pool_stride_d - pool_pad_d;
                const int end_md = start_md + pool_kernel_d;
                const int start_mh = ph * pool_stride_h - pool_pad_h;
                const int end_mh = start_mh + pool_kernel_h;
                const int start_mw = pw * pool_stride_w - pool_pad_w;
                const int end_mw = start_mw + pool_kernel_w;

                float max_val = -1e30f;
                bool valid_max = false;
                
                for (int md = start_md; md < end_md; md++) {
                    for (int mh = start_mh; mh < end_mh; mh++) {
                        for (int mw = start_mw; mw < end_mw; mw++) {
                            // Conv to input mapping
                            const int start_cd = md * conv_stride_d - conv_pad_d;
                            const int end_cd = start_cd + kernel_d;
                            const int start_ch = mh * conv_stride_h - conv_pad_h;
                            const int end_ch = start_ch + kernel_h;
                            const int start_cw = mw * conv_stride_w - conv_pad_w;
                            const int end_cw = start_cw + kernel_w;
                            
                            float conv_result = conv_bias[out_c_idx];
                            
                            for (int kd = 0; kd < kernel_d; kd++) {
                                const int in_d = start_cd + kd;
                                if (in_d < 0 || in_d >= in_depth) continue;
                                for (int kh = 0; kh < kernel_h; kh++) {
                                    const int in_h = start_ch + kh;
                                    if (in_h < 0 || in_h >= in_height) continue;
                                    for (int kw = 0; kw < kernel_w; kw++) {
                                        const int in_w = start_cw + kw;
                                        if (in_w < 0 || in_w >= in_width) continue;
                                        
                                        for (int ic = 0; ic < in_channels; ic++) {
                                            const int weight_idx = out_c_idx * (in_channels * kernel_d * kernel_h * kernel_w) +
                                                                   ic * (kernel_d * kernel_h * kernel_w) +
                                                                   kd * (kernel_h * kernel_w) +
                                                                   kh * kernel_w +
                                                                   kw;
                                            const int input_idx = index_to_position(out_n_idx, ic, in_d, in_h, in_w,
                                                                                    in_channels, in_depth, in_height, in_width);
                                            conv_result += input[input_idx] * conv_weight[weight_idx];
                                        }
                                    }
                                }
                            }
                            
                            const float normalized_val = conv_result / divisor;
                            if (!valid_max || normalized_val > max_val) {
                                max_val = normalized_val;
                                valid_max = true;
                            }
                        }
                    }
                }
                
                if (valid_max) {
                    sum += max_val;
                    count++;
                }
            }
        }
    }

    if (count > 0) {
        const float avg_val = sum / static_cast<float>(count);
        const float biased_val = avg_val + bias_data[out_c_idx];
        output[tid] = biased_val;
    } else {
        output[tid] = bias_data[out_c_idx];
    }
}

void launch_fused_op(
    const float* input,
    const float* conv_weight,
    const float* conv_bias,
    const float* bias_data,
    float* output,
    const int batch_size, const int in_channels, const int out_channels,
    const int in_depth, const int in_height, const int in_width,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int conv_stride_d, const int conv_stride_h, const int conv_stride_w,
    const int conv_pad_d, const int conv_pad_h, const int conv_pad_w,
    const int pool_kernel_d, const int pool_kernel_h, const int pool_kernel_w,
    const int pool_stride_d, const int pool_stride_h, const int pool_stride_w,
    const int pool_pad_d, const int pool_pad_h, const int pool_pad_w,
    const int out_depth, const int out_height, const int out_width,
    const int pool_out_depth, const int pool_out_height, const int pool_out_width,
    const int adaptive_out_depth, const int adaptive_out_height, const int adaptive_out_width,
    const float divisor,
    const int sum_dim
) {
    const int total_elements = batch_size * out_channels * adaptive_out_depth * adaptive_out_height * adaptive_out_width;
    const int threads_per_block = min(CUDA_MAX_THREADS_PER_BLOCK, total_elements);
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    fused_conv_pool_poolavg_add_sum_kernel<<<num_blocks, threads_per_block>>>(
        input, conv_weight, conv_bias, bias_data, output,
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        kernel_d, kernel_h, kernel_w,
        conv_stride_d, conv_stride_h, conv_stride_w,
        conv_pad_d, conv_pad_h, conv_pad_w,
        pool_kernel_d, pool_kernel_h, pool_kernel_w,
        pool_stride_d, pool_stride_h, pool_stride_w,
        pool_pad_d, pool_pad_h, pool_pad_w,
        out_depth, out_height, out_width,
        pool_out_depth, pool_out_height, pool_out_width,
        adaptive_out_depth, adaptive_out_height, adaptive_out_width,
        divisor,
        sum_dim
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void launch_fused_op(
    const float* input,
    const float* conv_weight,
    const float* conv_bias,
    const float* bias_data,
    float* output,
    const int batch_size, const int in_channels, const int out_channels,
    const int in_depth, const int in_height, const int in_width,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int conv_stride_d, const int conv_stride_h, const int conv_stride_w,
    const int conv_pad_d, const int conv_pad_h, const int conv_pad_w,
    const int pool_kernel_d, const int pool_kernel_h, const int pool_kernel_w,
    const int pool_stride_d, const int pool_stride_h, const int pool_stride_w,
    const int pool_pad_d, const int pool_pad_h, const int pool_pad_w,
    const int out_depth, const int out_height, const int out_width,
    const int pool_out_depth, const int pool_out_height, const int pool_out_width,
    const int adaptive_out_depth, const int adaptive_out_height, const int adaptive_out_width,
    const float divisor,
    const int sum_dim
);

torch::Tensor fused_conv_pool_poolavg_add_sum(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor bias,
    const std::vector<int64_t>& conv_stride,
    const std::vector<int64_t>& conv_padding,
    const std::vector<int64_t>& max_pool_kernel_size,
    const std::vector<int64_t>& max_pool_stride,
    const std::vector<int64_t>& max_pool_padding,
    float divisor,
    std::vector<int64_t> global_avg_pool_output_size,
    int64_t sum_dim
) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_depth = input.size(2);
    const auto in_height = input.size(3);
    const auto in_width = input.size(4);
    const auto out_channels = conv_weight.size(0);
    
    const auto kernel_d = conv_weight.size(2);
    const auto kernel_h = conv_weight.size(3);
    const auto kernel_w = conv_weight.size(4);
    
    // Conv output calculation
    const auto out_depth = (in_depth + 2 * conv_padding[0] - kernel_d) / conv_stride[0] + 1;
    const auto out_height = (in_height + 2 * conv_padding[1] - kernel_h) / conv_stride[1] + 1;
    const auto out_width = (in_width + 2 * conv_padding[2] - kernel_w) / conv_stride[2] + 1;

    // Max pool output calculation
    const auto pool_out_depth = (out_depth + 2 * max_pool_padding[0] - max_pool_kernel_size[0]) / max_pool_stride[0] + 1;
    const auto pool_out_height = (out_height + 2 * max_pool_padding[1] - max_pool_kernel_size[1]) / max_pool_stride[1] + 1;
    const auto pool_out_width = (out_width + 2 * max_pool_padding[2] - max_pool_kernel_size[2]) / max_pool_stride[2] + 1;

    auto output = torch::empty({batch_size, out_channels, global_avg_pool_output_size[0], global_avg_pool_output_size[1], global_avg_pool_output_size[2]}, 
                               torch::dtype(input.dtype()).device(input.device()));
    
    launch_fused_op(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        kernel_d, kernel_h, kernel_w,
        conv_stride[0], conv_stride[1], conv_stride[2],
        conv_padding[0], conv_padding[1], conv_padding[2],
        max_pool_kernel_size[0], max_pool_kernel_size[1], max_pool_kernel_size[2],
        max_pool_stride[0], max_pool_stride[1], max_pool_stride[2],
        max_pool_padding[0], max_pool_padding[1], max_pool_padding[2],
        out_depth, out_height, out_width,
        pool_out_depth, pool_out_height, pool_out_width,
        global_avg_pool_output_size[0], global_avg_pool_output_size[1], global_avg_pool_output_size[2],
        divisor,
        sum_dim
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_pool_poolavg_add_sum", &fused_conv_pool_poolavg_add_sum, "Fused conv, pool, poolavg, add, and sum operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op_ext',
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
    # Since dilation and groups are not used in the original code, we ignore them
    # Since ceil_mode and return_indices are not used for optimization, we ignore them
    return fused_ext.fused_conv_pool_poolavg_add_sum(
        x,
        conv_weight,
        conv_bias,
        bias,
        conv_stride,
        conv_padding,
        max_pool_kernel_size,
        max_pool_stride,
        max_pool_padding,
        divisor,
        global_avg_pool_output_size,
        sum_dim
    )

# Parameters matching original code
batch_size   = 128  
in_channels  = 8            
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
    return [torch.rand(batch_size, in_channels, depth, height, width, device='cuda')]
