# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_053107/code_1.py
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
#include <cub/cub.cuh>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void fused_model_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth,
    int in_height,
    int in_width,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int pool_kernel_d,
    int pool_kernel_h,
    int pool_kernel_w,
    int pool_stride_d,
    int pool_stride_h,
    int pool_stride_w,
    int out_depth,
    int out_height,
    int out_width,
    int pooled_depth,
    int pooled_height,
    int pooled_width,
    float divisor
) {
    // Each thread handles one output element
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * pooled_depth * pooled_height * pooled_width;
    
    if (tid >= total_outputs) return;
    
    // Decode output index
    int n = tid / (out_channels * pooled_depth * pooled_height * pooled_width);
    int oc = (tid / (pooled_depth * pooled_height * pooled_width)) % out_channels;
    int pd = (tid / (pooled_height * pooled_width)) % pooled_depth;
    int ph = (tid / pooled_width) % pooled_height;
    int pw = tid % pooled_width;

    // Map pooled coordinates back to conv output coordinates (before pooling)
    int start_d = pd * pool_stride_d;
    int start_h = ph * pool_stride_h;
    int start_w = pw * pool_stride_w;

    float max_val = -1e30f;
    bool first_iteration = true;

    // Iterate through the pooling window
    for (int kd = 0; kd < pool_kernel_d; ++kd) {
        for (int kh = 0; kh < pool_kernel_h; ++kh) {
            for (int kw = 0; kw < pool_kernel_w; ++kw) {
                int cd = start_d + kd;
                int ch = start_h + kh;
                int cw = start_w + kw;

                if (cd < 0 || cd >= out_depth || ch < 0 || ch >= out_height || cw < 0 || cw >= out_width) {
                    continue;
                }

                // Compute convolution at (cd, ch, cw)
                float conv_sum = 0.0f;
                
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kd_w = 0; kd_w < kernel_d; ++kd_w) {
                        for (int kh_w = 0; kh_w < kernel_h; ++kh_w) {
                            for (int kw_w = 0; kw_w < kernel_w; ++kw_w) {
                                int id = cd * stride_d - pad_d + kd_w * dilation_d;
                                int ih = ch * stride_h - pad_h + kh_w * dilation_h;
                                int iw = cw * stride_w - pad_w + kw_w * dilation_w;
                                
                                if (id >= 0 && id < in_depth && ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    float input_val = input[(((n * in_channels + ic) * in_depth + id) * in_height + ih) * in_width + iw];
                                    float weight_val = weight[(((oc * in_channels + ic) * kernel_d + kd_w) * kernel_h + kh_w) * kernel_w + kw_w];
                                    conv_sum += input_val * weight_val;
                                }
                            }
                        }
                    }
                }
                
                conv_sum += conv_bias[oc]; // Add bias
                float normalized_val = conv_sum / divisor;

                if (first_iteration || normalized_val > max_val) {
                    max_val = normalized_val;
                    first_iteration = false;
                }
            }
        }
    }

    if (!first_iteration) {
        // Add global bias and write to output
        output[tid] = max_val + bias[oc];
    }
}

__global__ void reduce_sum_kernel(const float* input, float* output, int batch_size, int out_channels, int spatial_size) {
    int oc = blockIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size) return;

    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float thread_data = 0.0f;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        int idx = ((batch_idx * out_channels + oc) * spatial_size) + i;
        thread_data += input[idx];
    }

    float aggregate = BlockReduce(temp_storage).Sum(thread_data);
    
    if (threadIdx.x == 0) {
        atomicAdd(&output[batch_idx * out_channels + oc], aggregate);
    }
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor bias,
    torch::Tensor intermediate_output,
    torch::Tensor output,
    float divisor,
    std::vector<int64_t> conv_stride,
    std::vector<int64_t> conv_padding,
    std::vector<int64_t> conv_dilation,
    int conv_groups,
    std::vector<int64_t> max_pool_kernel_size,
    std::vector<int64_t> max_pool_stride,
    std::vector<int64_t> max_pool_padding,
    std::vector<int64_t> max_pool_dilation,
    bool max_pool_ceil_mode,
    bool max_pool_return_indices
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);
    
    const int out_channels = weight.size(0);
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);

    const int stride_d = conv_stride[0];
    const int stride_h = conv_stride[1];
    const int stride_w = conv_stride[2];

    const int pad_d = conv_padding[0];
    const int pad_h = conv_padding[1];
    const int pad_w = conv_padding[2];

    const int dilation_d = conv_dilation[0];
    const int dilation_h = conv_dilation[1];
    const int dilation_w = conv_dilation[2];

    const int out_depth = (in_depth + 2 * pad_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    const int out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    const int pool_kernel_d = max_pool_kernel_size[0];
    const int pool_kernel_h = max_pool_kernel_size[1];
    const int pool_kernel_w = max_pool_kernel_size[2];

    const int pool_stride_d = max_pool_stride[0];
    const int pool_stride_h = max_pool_stride[1];
    const int pool_stride_w = max_pool_stride[2];

    const int pooled_depth = (out_depth - pool_kernel_d) / pool_stride_d + 1;
    const int pooled_height = (out_height - pool_kernel_h) / pool_stride_h + 1;
    const int pooled_width = (out_width - pool_kernel_w) / pool_stride_w + 1;

    const int total_outputs = batch_size * out_channels * pooled_depth * pooled_height * pooled_width;
    const int threads_per_block = 256;
    const int blocks = (total_outputs + threads_per_block - 1) / threads_per_block;

    fused_model_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        bias.data_ptr<float>(),
        intermediate_output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dilation_d,
        dilation_h,
        dilation_w,
        pool_kernel_d,
        pool_kernel_h,
        pool_kernel_w,
        pool_stride_d,
        pool_stride_h,
        pool_stride_w,
        out_depth,
        out_height,
        out_width,
        pooled_depth,
        pooled_height,
        pooled_width,
        divisor
    );

    // Reduction step
    dim3 reduce_blocks(out_channels, (batch_size + 15) / 16);
    dim3 reduce_threads(32, 16);
    reduce_sum_kernel<<<reduce_blocks, reduce_threads>>>(
        intermediate_output.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        pooled_depth * pooled_height * pooled_width
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
#include <vector>

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor bias,
    torch::Tensor intermediate_output,
    torch::Tensor output,
    float divisor,
    std::vector<int64_t> conv_stride,
    std::vector<int64_t> conv_padding,
    std::vector<int64_t> conv_dilation,
    int conv_groups,
    std::vector<int64_t> max_pool_kernel_size,
    std::vector<int64_t> max_pool_stride,
    std::vector<int64_t> max_pool_padding,
    std::vector<int64_t> max_pool_dilation,
    bool max_pool_ceil_mode,
    bool max_pool_return_indices
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused 3D Conv + Div + MaxPool + Bias + Sum");
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
    batch_size = x.shape[0]
    out_channels = conv_weight.shape[0]
    
    # Compute intermediate dimensions
    in_depth, in_height, in_width = x.shape[2], x.shape[3], x.shape[4]
    kernel_d, kernel_h, kernel_w = conv_weight.shape[2], conv_weight.shape[3], conv_weight.shape[4]
    
    stride_d, stride_h, stride_w = conv_stride[0], conv_stride[1], conv_stride[2]
    pad_d, pad_h, pad_w = conv_padding[0], conv_padding[1], conv_padding[2]
    dilation_d, dilation_h, dilation_w = conv_dilation[0], conv_dilation[1], conv_dilation[2]
    
    out_depth = (in_depth + 2 * pad_d - dilation_d * (kernel_d - 1) - 1) // stride_d + 1
    out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    pool_kernel_d, pool_kernel_h, pool_kernel_w = max_pool_kernel_size[0], max_pool_kernel_size[1], max_pool_kernel_size[2]
    pool_stride_d, pool_stride_h, pool_stride_w = max_pool_stride[0], max_pool_stride[1], max_pool_stride[2]
    
    pooled_depth = (out_depth - pool_kernel_d) // pool_stride_d + 1
    pooled_height = (out_height - pool_kernel_h) // pool_stride_h + 1
    pooled_width = (out_width - pool_kernel_w) // pool_stride_w + 1
    
    # Allocate intermediate and output tensors
    intermediate_output = torch.empty((batch_size, out_channels, pooled_depth, pooled_height, pooled_width), device=x.device, dtype=x.dtype)
    output = torch.empty((batch_size, out_channels), device=x.device, dtype=x.dtype)
    
    # Call the fused kernel
    fused_ext.fused_op(
        x, conv_weight, conv_bias, bias, intermediate_output, output, divisor,
        list(conv_stride), list(conv_padding), list(conv_dilation), conv_groups,
        list(max_pool_kernel_size), list(max_pool_stride), list(max_pool_padding),
        list(max_pool_dilation), max_pool_ceil_mode, max_pool_return_indices
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
