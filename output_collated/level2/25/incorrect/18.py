# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_083021/code_4.py
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

# CUDA kernel performing fused conv + channel-min + double-tanh
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out, int H, int W,
    int K, int stride, int padding
) {
    // Output dimensions
    int OH = (H + 2 * padding - K) / stride + 1;
    int OW = (W + 2 * padding - K) / stride + 1;

    // Thread and block indices
    int ow = blockIdx.x;
    int oh = blockIdx.y;
    int n = blockIdx.z;
    
    if (ow >= OW || oh >= OH) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Shared memory for partial min values
    extern __shared__ float s_min[];

    // Process channels in grid-stride loop
    float min_val = INFINITY;
    for (int co = tid; co < C_out; co += block_size) {
        float sum = bias[co];
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
        if (sum < min_val) min_val = sum;
    }

    // Warp-level reduction to find min across threads
    for (int offset = 16; offset > 0; offset /= 2) {
        float n_val = __shfl_down_sync(0xFFFFFFFF, min_val, offset);
        if (n_val < min_val) min_val = n_val;
    }

    // Store warp results in shared memory
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) {
        s_min[warp_id] = min_val;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        min_val = (lane_id < (block_size + 31) / 32) ? s_min[lane_id] : INFINITY;
        for (int offset = 1; offset < (block_size + 31) / 32; offset *= 2) {
            float n_val = __shfl_down_sync(0xFFFFFFFF, min_val, offset);
            if (n_val < min_val) min_val = n_val;
        }

        // Apply double tanh activation and write result
        if (lane_id == 0) {
            float tanh_val = tanhf(min_val);
            output[(n * OH + oh) * OW + ow] = tanhf(tanh_val);
        }
    }
}

void fused_conv_min_tanh_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int N, int C_in, int C_out, int H, int W,
    int K, int stride, int padding
) {
    int OH = (H + 2 * padding - K) / stride + 1;
    int OW = (W + 2 * padding - K) / stride + 1;

    dim3 grid(OW, OH, N);
    dim3 block(256);
    int shared_mem_size = ((256 + 31) / 32) * sizeof(float);

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
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int N, int C_in, int C_out, int H, int W,
    int K, int stride, int padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_tanh_forward", &fused_conv_min_tanh_forward, "Fused Conv-Min-Tanh forward pass");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_min_tanh',
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
    # Ensure inputs are on CUDA
    x = x.contiguous().cuda()
    conv_weight = conv_weight.contiguous().cuda()
    conv_bias = conv_bias.contiguous().cuda()
    
    # Extract dimensions
    N, C_in, H, W = x.shape
    C_out, _, K, _ = conv_weight.shape
    
    # Compute output dimensions
    OH = (H + 2 * conv_padding - K) // conv_stride + 1
    OW = (W + 2 * conv_padding - K) // conv_stride + 1
    
    # Allocate output tensor
    output = torch.empty((N, 1, OH, OW), device='cuda', dtype=torch.float32)
    
    # Launch fused kernel
    fused_ext.fused_conv_min_tanh_forward(
        x, conv_weight, conv_bias, output,
        N, C_in, C_out, H, W, K, conv_stride, conv_padding
    )
    
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
