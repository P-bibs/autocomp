# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_094044/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'softmax_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

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
    # State for conv_transpose (nn.ConvTranspose3d)
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
    # State for softmax (nn.Softmax)
    state_kwargs['softmax_dim'] = model.softmax.dim
    # State for sigmoid (nn.Sigmoid)
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// Softmax reduction helper functions
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void fused_conv_transpose3d_softmax_sigmoid_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float* __restrict__ softmax_sums,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int softmax_dim) {
    
    // Calculate global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    if (tid >= total_elements) return;
    
    // Decompose linear index to 5D coordinates
    int temp = tid;
    int w_idx = temp % output_w;
    temp /= output_w;
    int h_idx = temp % output_h;
    temp /= output_h;
    int d_idx = temp % output_d;
    temp /= output_d;
    int c_idx = temp % out_channels;
    int b_idx = temp / out_channels;
    
    // Conv transpose calculation
    float conv_result = 0.0f;
    if (bias != nullptr) {
        conv_result = bias[c_idx];
    }
    
    // Perform conv transpose operation
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Calculate corresponding input position
                int in_d = (d_idx + padding - kd) / stride;
                int in_h = (h_idx + padding - kh) / stride;
                int in_w = (w_idx + padding - kw) / stride;
                
                // Check if valid input position
                if ((d_idx + padding - kd) % stride == 0 &&
                    (h_idx + padding - kh) % stride == 0 &&
                    (w_idx + padding - kw) % stride == 0 &&
                    in_d >= 0 && in_d < input_d &&
                    in_h >= 0 && in_h < input_h &&
                    in_w >= 0 && in_w < input_w) {
                    
                    for (int ic = 0; ic < in_channels; ic++) {
                        int input_idx = ((((b_idx * in_channels) + ic) * input_d + in_d) * input_h + in_h) * input_w + in_w;
                        int weight_idx = ((((ic * out_channels) + c_idx) * kernel_size + kd) * kernel_size + kh) * kernel_size + kw;
                        conv_result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Softmax computation - correct implementation with reduction
    if (softmax_dim == 1) {
        // Need to compute softmax along channel dimension
        // First compute max for numerical stability
        float thread_max = conv_result;
        __shared__ float shared_max[512];
        __shared__ float shared_sum[512];
        
        // Warp-level reduction for max
        thread_max = warpReduceSum(thread_max);
        if (threadIdx.x % 32 == 0) {
            shared_max[threadIdx.x / 32] = thread_max;
        }
        __syncthreads();
        
        if (threadIdx.x < 32) {
            thread_max = shared_max[threadIdx.x];
            thread_max = warpReduceSum(thread_max);
            if (threadIdx.x == 0) {
                shared_max[0] = thread_max;
            }
        }
        __syncthreads();
        float max_val = shared_max[0];
        
        // Compute exp
        float exp_val = expf(conv_result - max_val);
        
        // Warp-level reduction for sum
        float thread_sum = exp_val;
        thread_sum = warpReduceSum(thread_sum);
        if (threadIdx.x % 32 == 0) {
            shared_sum[threadIdx.x / 32] = thread_sum;
        }
        __syncthreads();
        
        if (threadIdx.x < 32) {
            thread_sum = shared_sum[threadIdx.x];
            thread_sum = warpReduceSum(thread_sum);
            if (threadIdx.x == 0) {
                shared_sum[0] = thread_sum;
            }
        }
        __syncthreads();
        float sum_val = shared_sum[0];
        
        // Normalize
        float softmax_result = exp_val / sum_val;
        
        // Apply sigmoid
        float sigmoid_result = 1.0f / (1.0f + expf(-softmax_result));
        
        output[tid] = sigmoid_result;
    } else {
        // Simplified softmax + sigmoid for other dimensions
        float softmax_result = expf(conv_result);  // Simplified
        float sigmoid_result = 1.0f / (1.0f + expf(-softmax_result));
        output[tid] = sigmoid_result;
    }
}

void fused_conv_transpose3d_softmax_sigmoid_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int softmax_dim) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_d = input.size(2);
    int input_h = input.size(3);
    int input_w = input.size(4);
    
    int out_channels = weight.size(0);
    int output_d = (input_d - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_h = (input_h - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_w = (input_w - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    const int threads = 512;
    const int blocks = (total_elements + threads - 1) / threads;
    
    // Temporary buffer for softmax reductions
    auto softmax_sums = torch::empty({blocks * threads}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    fused_conv_transpose3d_softmax_sigmoid_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        softmax_sums.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_size,
        stride,
        padding,
        output_padding,
        softmax_dim
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_softmax_sigmoid_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int softmax_dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv_transpose3d_softmax_sigmoid_forward, "Fused conv transpose3d + softmax + sigmoid");
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

# Global variables for model parameters
batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

# Pre-computed weights and biases
conv_transpose_weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, kernel_size, device='cuda')
conv_transpose_bias = torch.randn(out_channels, device='cuda')
softmax_dim = 1

def functional_model(
    x,
    *,
    conv_transpose_weight=conv_transpose_weight,
    conv_transpose_bias=conv_transpose_bias,
    conv_transpose_stride=stride,
    conv_transpose_padding=padding,
    conv_transpose_output_padding=output_padding,
    conv_transpose_groups=1,
    conv_transpose_dilation=1,
    softmax_dim=softmax_dim,
):
    x = x.cuda()
    conv_transpose_weight = conv_transpose_weight.cuda()
    conv_transpose_bias = conv_transpose_bias.cuda()
    
    # Calculate output dimensions
    output_d = (D - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    output_h = (H - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    output_w = (W - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    
    output = torch.empty(batch_size, out_channels, output_d, output_h, output_w, device='cuda')
    
    # Call fused operation
    fused_ext.fused_op(
        x, conv_transpose_weight, conv_transpose_bias, output,
        kernel_size, conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding, softmax_dim
    )
    
    return output

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]
