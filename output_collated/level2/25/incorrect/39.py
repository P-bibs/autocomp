# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_085904/code_0.py
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

# --- Optimized CUDA Kernel ---
cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#define TILE_DIM 16
#define CHANNEL_TILE 32

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out,
    int H, int W, int K,
    int stride, int padding
) {
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.z;

    int OH = (H + 2 * padding - K) / stride + 1;
    int OW = (W + 2 * padding - K) / stride + 1;

    if (oh >= OH || ow >= OW) return;

    extern __shared__ float shared_mem[];
    float* s_x = shared_mem;
    float* s_w = &shared_mem[TILE_DIM*TILE_DIM*C_in];

    // Process output channels in tiles
    for (int co_start = 0; co_start < C_out; co_start += CHANNEL_TILE) {
        int c_end = min(co_start + CHANNEL_TILE, C_out);
        
        float result[CHANNEL_TILE];  // Local registers for accumulation

        // Convolve and accumulate
        for (int co = co_start; co < c_end; ++co) {
            int local_idx = co - co_start;
            result[local_idx] = bias[co];

            for (int ci = 0; ci < C_in; ++ci) {
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        int ih = oh * stride + kh - padding;
                        int iw = ow * stride + kw - padding;
                        float val_x = 0.f;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            val_x = x[((n * C_in + ci) * H + ih) * W + iw];
                        }
                        float val_w = weight[(((co * C_in + ci) * K + kh) * K + kw)];
                        result[local_idx] += val_x * val_w;
                    }
                }
            }
        }

        // Reduce across channel dimension (using simple intra-block reduction)
        __shared__ float min_buffer[CHANNEL_TILE];
        float thread_min = INFINITY;

        for (int co = co_start; co < c_end; ++co) {
            int local_idx = co - co_start;
            thread_min = fminf(thread_min, result[local_idx]);
        }

        if (threadIdx.x < CHANNEL_TILE) {
            min_buffer[threadIdx.x] = thread_min;
        }
        __syncthreads();

        // Intra-warp reduction
        for (int stride = 1; stride < min(32, CHANNEL_TILE); stride *= 2) {
            if ((threadIdx.x % (2 * stride)) == 0 && (threadIdx.x + stride) < CHANNEL_TILE) {
                min_buffer[threadIdx.x] = fminf(min_buffer[threadIdx.x], min_buffer[threadIdx.x + stride]);
            }
            __syncthreads();
        }

        // Double Tanh activation
        if (threadIdx.x == 0) {
            float val = min_buffer[0];
            val = tanhf(val);
            val = tanhf(val);
            output[((n * 1 + 0) * OH + oh) * OW + ow] = val;
        }
        __syncthreads();
    }
}
'''

# --- C++ Binding ---
cpp_source = r'''
#include <torch/extension.h>
#include <vector>

void launch_fused_kernel(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                         torch::Tensor out, int N, int C_in, int C_out,
                         int H, int W, int K, int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_kernel", &launch_fused_kernel, "Fused Conv-Min-Tanh kernel");
}
'''

# --- Launch Wrapper ---
def compile_fused_kernel():
    return load_inline(
        name='fused_kernel',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        with_cuda=True
    )

# Global extension instance
_fused_ext = None

def get_fused_kernel():
    global _fused_ext
    if _fused_ext is None:
        _fused_ext = compile_fused_kernel()
    return _fused_ext

# --- Fused Operation Definition ---
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
    assert conv_dilation == 1 and conv_groups == 1, "Unsupported parameters"

    ext = get_fused_kernel()
    
    N, C_in, H, W = x.shape
    C_out, _, K, _ = conv_weight.shape

    OH = (H + 2 * conv_padding - K) // conv_stride + 1
    OW = (W + 2 * conv_padding - K) // conv_stride + 1

    out = torch.empty((N, 1, OH, OW), device=x.device, dtype=x.dtype)

    block_dim = (16, 16, 1)
    grid_dim = (
        (OW + block_dim[0] - 1) // block_dim[0],
        (OH + block_dim[1] - 1) // block_dim[1],
        N
    )

    # Dynamically compute required shared memory size
    shared_mem_size = TILE_DIM * TILE_DIM * C_in * sizeof(float) + CHANNEL_TILE * sizeof(float)

    ext.launch_fused_kernel(
        x.contiguous(), 
        conv_weight.contiguous(), 
        conv_bias.contiguous(), 
        out,
        N, C_in, C_out,
        H, W, K,
        conv_stride, conv_padding
    )
    return out

# --- Input Definitions for Evaluation ---
batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
