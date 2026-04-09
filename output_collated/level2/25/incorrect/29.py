# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_084802/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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
    # State for conv (nn.Conv2d)
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

# CUDA kernel that fuses convolution, channel-min reduction, and double tanh
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 16

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out, int H, int W, int K,
    int stride, int padding) {

    // Output dimensions
    int OH = (H + 2 * padding - K) / stride + 1;
    int OW = (W + 2 * padding - K) / stride + 1;
    
    // Thread and block indices
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (oh >= OH || ow >= OW || n >= N) return;
    
    // Shared memory for storing intermediate min values
    extern __shared__ float shared_min[];
    float* min_vals = shared_min;
    
    // Initialize minimum value to a large number
    float min_val = INFINITY;
    
    // Iterate through all output channels to find minimum
    for (int co = 0; co < C_out; ++co) {
        float sum = bias[co];
        
        // Perform convolution for this output channel
        for (int ci = 0; ci < C_in; ++ci) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int ih = oh * stride + kh - padding;
                    int iw = ow * stride + kw - padding;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        sum += x[((n * C_in + ci) * H + ih) * W + iw] * 
                               weight[(((co * C_in + ci) * K + kh) * K + kw)];
                    }
                }
            }
        }
        
        // Update minimum value across channels
        min_val = fminf(min_val, sum);
    }
    
    // Store min value in shared memory
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    min_vals[tid] = min_val;
    __syncthreads();
    
    // Reduce within warp to find final minimum
    for (int stride = 8; stride > 0; stride >>= 1) {
        if (tid < stride) {
            min_vals[tid] = fminf(min_vals[tid], min_vals[tid + stride]);
        }
        __syncthreads();
    }
    
    // Apply double tanh activation
    float intermediate = tanhf(min_vals[0]);
    float final_result = tanhf(intermediate);
    
    // Write final result to output
    output[((n * 1) * OH + oh) * OW + ow] = final_result;
}

void fused_conv_min_tanh_forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding) {
    
    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int C_out = weight.size(0);
    int K = weight.size(2);
    
    int OH = (H + 2 * padding - K) / stride + 1;
    int OW = (W + 2 * padding - K) / stride + 1;
    
    // Grid and block dimensions
    dim3 block(16, 16, 1);
    dim3 grid((OW + block.x - 1) / block.x, 
              (OH + block.y - 1) / block.y, 
              (N + block.z - 1) / block.z);
    
    // Shared memory size for min reduction
    size_t shared_mem_size = block.x * block.y * sizeof(float);
    
    fused_conv_min_tanh_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out, H, W, K, stride, padding
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_min_tanh_forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_tanh", &fused_conv_min_tanh_forward, "Fused Conv-Min-Tanh forward");
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
):
    # Validate input parameters
    if conv_dilation != (1, 1) or conv_groups != 1:
        raise NotImplementedError("Only dilation=1 and groups=1 are supported")
    
    # Create output tensor with correct dimensions
    N, C_in, H, W = x.shape
    C_out, _, K, _ = conv_weight.shape
    OH = (H + 2 * conv_padding - K) // conv_stride + 1
    OW = (W + 2 * conv_padding - K) // conv_stride + 1
    
    output = torch.empty(N, 1, OH, OW, device=x.device, dtype=x.dtype)
    
    # Call our fused CUDA kernel
    fused_ext.fused_conv_min_tanh(x, conv_weight, conv_bias, output, conv_stride, conv_padding)
    
    return output

# Input parameters
batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
