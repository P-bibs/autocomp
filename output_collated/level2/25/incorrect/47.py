# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_090933/code_2.py
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

# CUDA Kernel: Fused 2D Conv + Channel-Min + Double Tanh with Shared Memory Tiling
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 16
#define MAX_CHANNELS 256

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
    float* __restrict__ out, int N, int C_in, int C_out, int H, int W, int K, int stride, int padding) {

    extern __shared__ float shared_x[];
    
    int OH = (H + 2 * padding - K) / stride + 1;
    int OW = (W + 2 * padding - K) / stride + 1;
    
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;
    
    if (oh >= OH || ow >= OW) return;

    // Each thread computes one output element for all output channels
    __shared__ float local_sums[MAX_CHANNELS];
    
    for (int co = 0; co < C_out; ++co) {
        float sum = b[co];
        
        // Load input tile to shared memory
        for (int ci = 0; ci < C_in; ++ci) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int ih = oh * stride + kh - padding;
                    int iw = ow * stride + kw - padding;
                    float val = 0.0f;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        val = x[((n * C_in + ci) * H + ih) * W + iw];
                    }
                    sum += val * w[(((co * C_in + ci) * K + kh) * K + kw)];
                }
            }
        }
        
        local_sums[co] = sum;
    }
    
    __syncthreads();
    
    // Channel-wise min reduction
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float min_val = local_sums[0];
        for (int co = 1; co < C_out; ++co) {
            if (local_sums[co] < min_val) {
                min_val = local_sums[co];
            }
        }
        
        // Double Tanh activation
        float res = tanhf(min_val);
        res = tanhf(res);
        
        out[((n * 1) * OH + oh) * OW + ow] = res;
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out, int stride, int padding) {
    const int N = x.size(0);
    const int C_in = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int C_out = w.size(0);
    const int K = w.size(2);
    
    const int OH = (H + 2 * padding - K) / stride + 1;
    const int OW = (W + 2 * padding - K) / stride + 1;

    dim3 block(16, 16);
    dim3 grid((OW + block.x - 1) / block.x, (OH + block.y - 1) / block.y, N);
    
    size_t shared_mem_size = 0; // We're not actually using shared memory for input in this version
    
    fused_conv_min_tanh_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
        out.data_ptr<float>(), N, C_in, C_out, H, W, K, stride, padding);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out, int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv-Min-Tanh with Shared Memory");
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
    # Validate dilation and groups (simplified implementation assumes default values)
    if conv_dilation != 1 or conv_groups != 1:
        raise NotImplementedError("Only conv_dilation=1 and conv_groups=1 are supported")
    
    N, C_in, H, W = x.shape
    C_out, _, K, _ = conv_weight.shape
    
    # Calculate output dimensions
    OH = (H + 2 * conv_padding - K) // conv_stride + 1
    OW = (W + 2 * conv_padding - K) // conv_stride + 1
    
    # Create output tensor
    out = torch.empty((N, 1, OH, OW), device='cuda', dtype=x.dtype)
    
    # Call the custom CUDA kernel
    fused_ext.fused_op(x, conv_weight, conv_bias, out, conv_stride, conv_padding)
    
    return out

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
