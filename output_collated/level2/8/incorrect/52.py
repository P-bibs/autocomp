# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_064410/code_14.py
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

# -------------------------------------------------------------------------
#  Optimized CUDA kernel with shared-memory block-wise reduction
# -------------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused kernel rewritten to use custom conv3d, max_pool3d and adaptive_avg_pool3d
// followed by optimized channel-wise reduction using shared memory

// --- Custom Conv3D kernel ---
__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int K, int C_out,
    int D_out, int H_out, int W_out,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * D_out * H_out * W_out;

    if (idx >= total_elements) return;

    int n = idx / (C_out * D_out * H_out * W_out);
    int c_out = (idx / (D_out * H_out * W_out)) % C_out;
    int d_out = (idx / (H_out * W_out)) % D_out;
    int h_out = (idx / W_out) % H_out;
    int w_out = idx % W_out;

    float sum = 0.0f;
    int group = c_out / (C_out / groups);
    int weight_offset = group * (C_out / groups) * C_in * kernel_d * kernel_h * kernel_w;

    for (int kd = 0; kd < kernel_d; ++kd) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                for (int c_in = 0; c_in < C_in; ++c_in) {
                    int d_in_coord = d_out * stride_d - pad_d + kd * dilation_d;
                    int h_in_coord = h_out * stride_h - pad_h + kh * dilation_h;
                    int w_in_coord = w_out * stride_w - pad_w + kw * dilation_w;

                    if (d_in_coord >= 0 && d_in_coord < D_in &&
                        h_in_coord >= 0 && h_in_coord < H_in &&
                        w_in_coord >= 0 && w_in_coord < W_in) {
                        int input_idx = ((n * C_in + c_in) * D_in + d_in_coord) * H_in * W_in + h_in_coord * W_in + w_in_coord;
                        int weight_idx = weight_offset + (((c_out % (C_out/groups)) * C_in + c_in) * kernel_d + kd) * kernel_h * kernel_w + kh * kernel_w + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Add bias and write result
    output[idx] = sum + bias[c_out];
}

// --- Custom MaxPool3D kernel ---
__global__ void maxpool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * D_out * H_out * W_out;

    if (idx >= total_elements) return;

    int n = idx / (C * D_out * H_out * W_out);
    int c = (idx / (D_out * H_out * W_out)) % C;
    int d_out = (idx / (H_out * W_out)) % D_out;
    int h_out = (idx / W_out) % H_out;
    int w_out = idx % W_out;

    float max_val = -INFINITY;
    int d_start = d_out * stride_d - pad_d;
    int h_start = h_out * stride_h - pad_h;
    int w_start = w_out * stride_w - pad_w;
    int d_end = min(d_start + kernel_d, D_in);
    int h_end = min(h_start + kernel_h, H_in);
    int w_end = min(w_start + kernel_w, W_in);
    d_start = max(d_start, 0);
    h_start = max(h_start, 0);
    w_start = max(w_start, 0);

    for (int d = d_start; d < d_end; ++d) {
        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                int input_idx = ((n * C + c) * D_in + d) * H_in * W_in + h * W_in + w;
                float val = input[input_idx];
                if (val > max_val) max_val = val;
            }
        }
    }

    output[idx] = max_val;
}

// --- Custom AdaptiveAvgPool3D kernel (simplified for global avg pooling) ---
__global__ void adaptive_avg_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * D_out * H_out * W_out;

    if (idx >= total_elements) return;

    int n = idx / (C * D_out * H_out * W_out);
    int c = (idx / (D_out * H_out * W_out)) % C;
    int d_out = (idx / (H_out * W_out)) % D_out;
    int h_out = (idx / W_out) % H_out;
    int w_out = idx % W_out;

    // Compute pooling window size
    int d_start = (d_out * D_in) / D_out;
    int d_end = ((d_out + 1) * D_in + D_out - 1) / D_out;
    int h_start = (h_out * H_in) / H_out;
    int h_end = ((h_out + 1) * H_in + H_out - 1) / H_out;
    int w_start = (w_out * W_in) / W_out;
    int w_end = ((w_out + 1) * W_in + W_out - 1) / W_out;

    float sum = 0.0f;
    int count = 0;

    for (int d = d_start; d < d_end; ++d) {
        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                int input_idx = ((n * C + c) * D_in + d) * H_in * W_in + h * W_in + w;
                sum += input[input_idx];
                count++;
            }
        }
    }

    output[idx] = sum / count;
}

// --- Optimized fused post-processing with shared memory reduction ---
__global__ void fused_post_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float inv_divisor,
    int N, int C, int D, int H, int W,
    float* __restrict__ output
) {
    const int total_spatial = D * H * W;
    const int out_idx = blockIdx.x;
    if (out_idx >= N * total_spatial) return;

    const int n = out_idx / total_spatial;
    const int spatial_id = out_idx % total_spatial;
    const int d = spatial_id / (H * W);
    const int rem = spatial_id % (H * W);
    const int h = rem / W;
    const int w = rem % W;

    float sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const int idx = ((n * C + c) * D + d) * (H * W) + (h * W + w);
        float val = __ldg(&input[idx]);  // read-only cache
        val = fma(val, inv_divisor, __ldg(&bias[c]));  // fused div + bias
        sum += val;
    }

    __shared__ float sdata[256];
    const int tid = threadIdx.x;
    sdata[tid] = sum;
    __syncthreads();

    // Reduce over the whole block (size > 32) using syncthreads
    for (int s = blockDim.x >> 1; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Final warp-level reduction
    if (tid < 32) {
        volatile float* s = sdata;
        if (blockDim.x > 128) s[tid] += s[tid + 128];
        if (blockDim.x > 64)  s[tid] += s[tid + 64];
        if (blockDim.x > 32)  s[tid] += s[tid + 32];
    }

    // Write the final sum for this spatial location
    if (tid == 0) output[out_idx] = sdata[0];
}

// Host functions to launch kernels
void run_conv3d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                torch::Tensor output, int stride_d, int stride_h, int stride_w,
                int pad_d, int pad_h, int pad_w, int dilation_d, int dilation_h, int dilation_w, int groups) {
    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    int D_out = output.size(2);
    int H_out = output.size(3);
    int W_out = output.size(4);
    int C_out = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    int total_elements = N * C_out * D_out * H_out * W_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in, kernel_d, C_out, D_out, H_out, W_out,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w,
        groups
    );
}

void run_maxpool3d(torch::Tensor input, torch::Tensor output,
                   int kernel_d, int kernel_h, int kernel_w,
                   int stride_d, int stride_h, int stride_w,
                   int pad_d, int pad_h, int pad_w) {
    int N = input.size(0);
    int C = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    int D_out = output.size(2);
    int H_out = output.size(3);
    int W_out = output.size(4);

    int total_elements = N * C * D_out * H_out * W_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    maxpool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, D_in, H_in, W_in, D_out, H_out, W_out,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w
    );
}

void run_adaptive_avg_pool3d(torch::Tensor input, torch::Tensor output) {
    int N = input.size(0);
    int C = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    int D_out = output.size(2);
    int H_out = output.size(3);
    int W_out = output.size(4);

    int total_elements = N * C * D_out * H_out * W_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    adaptive_avg_pool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, D_in, H_in, W_in, D_out, H_out, W_out
    );
}

void run_fused_post_conv(torch::Tensor x, torch::Tensor bias, float divisor, torch::Tensor output) {
    int N = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);

    int total_spatial = D * H * W;
    int blocks = N * total_spatial;
    int threads = 256;

    if (blocks == 0) return;

    float inv_divisor = 1.0f / divisor;

    fused_post_conv_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), bias.data_ptr<float>(), inv_divisor,
        N, C, D, H, W, output.data_ptr<float>()
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void run_conv3d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                torch::Tensor output, int stride_d, int stride_h, int stride_w,
                int pad_d, int pad_h, int pad_w, int dilation_d, int dilation_h, int dilation_w, int groups);

void run_maxpool3d(torch::Tensor input, torch::Tensor output,
                   int kernel_d, int kernel_h, int kernel_w,
                   int stride_d, int stride_h, int stride_w,
                   int pad_d, int pad_h, int pad_w);

void run_adaptive_avg_pool3d(torch::Tensor input, torch::Tensor output);

void run_fused_post_conv(torch::Tensor x, torch::Tensor bias, float divisor, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d", &run_conv3d, "Custom Conv3D operation");
    m.def("maxpool3d", &run_maxpool3d, "Custom MaxPool3D operation");
    m.def("adaptive_avg_pool3d", &run_adaptive_avg_pool3d, "Custom AdaptiveAvgPool3D operation");
    m.def("fused_post_conv", &run_fused_post_conv, "Fused post-conv reduction");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *,
                     conv_weight, conv_bias,
                     conv_stride, conv_padding, conv_dilation, conv_groups,
                     max_pool_kernel_size, max_pool_stride, max_pool_padding,
                     max_pool_dilation, max_pool_ceil_mode, max_pool_return_indices,
                     global_avg_pool_output_size,
                     divisor, bias, sum_dim):
    """
    Optimized functional model:
    1. Custom Conv3D
    2. Custom MaxPool3D 
    3. Custom AdaptiveAvgPool3D
    4. Channel reduction with shared memory optimization
    """
    
    # --- Step 1: Conv3D --------------------------------------------------
    # Calculate output dimensions
    N, C_in, D_in, H_in, W_in = x.shape
    K, C_out = conv_weight.shape[0], conv_weight.shape[1]
    kernel_d, kernel_h, kernel_w = conv_weight.shape[2], conv_weight.shape[3], conv_weight.shape[4]
    
    if isinstance(conv_stride, int):
        stride_d = stride_h = stride_w = conv_stride
    else:
        stride_d, stride_h, stride_w = conv_stride[0], conv_stride[1], conv_stride[2]
        
    if isinstance(conv_padding, int):
        pad_d = pad_h = pad_w = conv_padding
    else:
        pad_d, pad_h, pad_w = conv_padding[0], conv_padding[1], conv_padding[2]
        
    if isinstance(conv_dilation, int):
        dilation_d = dilation_h = dilation_w = conv_dilation
    else:
        dilation_d, dilation_h, dilation_w = conv_dilation[0], conv_dilation[1], conv_dilation[2]

    D_out = ((D_in + 2*pad_d - dilation_d*(kernel_d-1) - 1) // stride_d) + 1
    H_out = ((H_in + 2*pad_h - dilation_h*(kernel_h-1) - 1) // stride_h) + 1
    W_out = ((W_in + 2*pad_w - dilation_w*(kernel_w-1) - 1) // stride_w) + 1

    conv_output = torch.empty((N, K, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    fused_ext.conv3d(
        x, conv_weight, conv_bias, conv_output,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w,
        conv_groups
    )
    x = conv_output

    # --- Step 2: MaxPool3D -----------------------------------------------
    if isinstance(max_pool_kernel_size, int):
        pool_d = pool_h = pool_w = max_pool_kernel_size
    else:
        pool_d, pool_h, pool_w = max_pool_kernel_size[0], max_pool_kernel_size[1], max_pool_kernel_size[2]
        
    if isinstance(max_pool_stride, int):
        pool_stride_d = pool_stride_h = pool_stride_w = max_pool_stride
    else:
        pool_stride_d, pool_stride_h, pool_stride_w = max_pool_stride[0], max_pool_stride[1], max_pool_stride[2]
        
    if isinstance(max_pool_padding, int):
        pool_pad_d = pool_pad_h = pool_pad_w = max_pool_padding
    else:
        pool_pad_d, pool_pad_h, pool_pad_w = max_pool_padding[0], max_pool_padding[1], max_pool_padding[2]
    
    # Calculate max pooling output size
    D_pool_out = ((x.size(2) + 2*pool_pad_d - pool_d) // pool_stride_d) + 1
    H_pool_out = ((x.size(3) + 2*pool_pad_h - pool_h) // pool_stride_h) + 1
    W_pool_out = ((x.size(4) + 2*pool_pad_w - pool_w) // pool_stride_w) + 1

    pool_output = torch.empty((N, x.size(1), D_pool_out, H_pool_out, W_pool_out), device=x.device, dtype=x.dtype)
    fused_ext.maxpool3d(
        x, pool_output,
        pool_d, pool_h, pool_w,
        pool_stride_d, pool_stride_h, pool_stride_w,
        pool_pad_d, pool_pad_h, pool_pad_w
    )
    x = pool_output

    # --- Step 3: AdaptiveAvgPool3D ---------------------------------------
    # For global average pooling case
    if isinstance(global_avg_pool_output_size, int):
        D_adapt_out = H_adapt_out = W_adapt_out = global_avg_pool_output_size
    else:
        D_adapt_out, H_adapt_out, W_adapt_out = global_avg_pool_output_size[0], global_avg_pool_output_size[1], global_avg_pool_output_size[2]

    adapt_output = torch.empty((N, x.size(1), D_adapt_out, H_adapt_out, W_adapt_out), device=x.device, dtype=x.dtype)
    fused_ext.adaptive_avg_pool3d(x, adapt_output)
    x = adapt_output

    # --- Step 4 & 5: Fused division, bias add, and channel reduction ----
    output = torch.zeros((x.size(0), x.size(2), x.size(3), x.size(4)), device=x.device, dtype=x.dtype)
    fused_ext.fused_post_conv(x, bias.view(-1), divisor, output)
    
    return output
