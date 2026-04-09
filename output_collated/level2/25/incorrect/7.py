# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_082041/code_4.py
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

# CUDA kernel implementing fused convolution with Implicit GEMM, shared memory tiling,
# channel-min reduction, and double tanh activation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 16
#define THREADS_PER_BLOCK 256

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out, int H, int W, int K,
    int stride, int padding) {
    
    // Shared memory for input tile (with halo)
    extern __shared__ float shared_x[];
    
    int OH = (H + 2 * padding - K) / stride + 1;
    int OW = (W + 2 * padding - K) / stride + 1;
    
    // Block and thread indices
    int block_oh = blockIdx.y * TILE_SIZE;
    int block_ow = blockIdx.x * TILE_SIZE;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int total_threads = blockDim.x * blockDim.y;
    
    // Load input data to shared memory
    // Size of input tile with padding: (TILE_SIZE + K - 1) x (TILE_SIZE + K - 1) x C_in
    int tile_h = TILE_SIZE + K - 1;
    int tile_w = TILE_SIZE + K - 1;
    
    // Each thread loads multiple elements to shared memory
    for (int i = tid; i < tile_h * tile_w * C_in; i += total_threads) {
        int ci = i % C_in;
        int tmp = i / C_in;
        int tw = tmp % tile_w;
        int th = tmp / tile_w;
        
        int ih = block_oh + th - padding;
        int iw = block_ow + tw - padding;
        
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            shared_x[i] = x[(((blockIdx.z * C_in) + ci) * H + ih) * W + iw];
        } else {
            shared_x[i] = 0.0f;
        }
    }
    __syncthreads();
    
    // Process output pixels in the tile
    for (int oh_offset = threadIdx.y; oh_offset < TILE_SIZE && (block_oh + oh_offset) < OH; oh_offset += blockDim.y) {
        for (int ow_offset = threadIdx.x; ow_offset < TILE_SIZE && (block_ow + ow_offset) < OW; ow_offset += blockDim.x) {
            int oh = block_oh + oh_offset;
            int ow = block_ow + ow_offset;
            
            // For each output channel, compute convolution result
            for (int co = 0; co < C_out; co++) {
                float sum = bias[co];
                
                // Perform convolution
                for (int ci = 0; ci < C_in; ci++) {
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            int ih_local = oh_offset + kh;
                            int iw_local = ow_offset + kw;
                            
                            sum += shared_x[((ci * tile_h + ih_local) * tile_w + iw_local)] * 
                                   weight[(((co * C_in + ci) * K + kh) * K + kw)];
                        }
                    }
                }
                
                // Apply first tanh
                sum = tanhf(sum);
                
                // Channel-wise min reduction (using shared memory for temporary storage)
                // In this simplified version, we do a simple reduction per thread
                __shared__ float min_vals[TILE_SIZE * TILE_SIZE * 32]; // Enough space for block
                int local_idx = oh_offset * TILE_SIZE + ow_offset;
                min_vals[local_idx * 32 + co % 32] = sum;
                __syncthreads();
                
                // Simple min reduction within warp
                for (int stride = 16; stride > 0; stride /= 2) {
                    if (co < stride && co + stride < C_out) {
                        float other = min_vals[local_idx * 32 + (co + stride) % 32];
                        if (other < min_vals[local_idx * 32 + co % 32]) {
                            min_vals[local_idx * 32 + co % 32] = other;
                        }
                    }
                    __syncthreads();
                }
                
                // Apply second tanh and write to output
                if (co == 0) {
                    float min_val = min_vals[local_idx * 32];
                    min_val = tanhf(min_val);
                    output[(((blockIdx.z * 1) + 0) * OH + oh) * OW + ow] = min_val;
                }
            }
        }
    }
}

void fused_conv_min_tanh_forward(
    const at::Tensor x,
    const at::Tensor weight,
    const at::Tensor bias,
    at::Tensor output,
    int N, int C_in, int C_out, int H, int W, int K,
    int stride, int padding) {
    
    int OH = (H + 2 * padding - K) / stride + 1;
    int OW = (W + 2 * padding - K) / stride + 1;
    
    // Grid dimensions
    dim3 grid_dim(
        (OW + TILE_SIZE - 1) / TILE_SIZE,
        (OH + TILE_SIZE - 1) / TILE_SIZE,
        N
    );
    
    // Block dimensions (32x8 = 256 threads)
    dim3 block_dim(32, 8);
    
    // Shared memory size
    int tile_h = TILE_SIZE + K - 1;
    int tile_w = TILE_SIZE + K - 1;
    size_t shared_mem_size = tile_h * tile_w * C_in * sizeof(float);
    
    fused_conv_min_tanh_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out, H, W, K, stride, padding
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_min_tanh_forward(
    const at::Tensor x,
    const at::Tensor weight,
    const at::Tensor bias,
    at::Tensor output,
    int N, int C_in, int C_out, int H, int W, int K,
    int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_tanh_forward", &fused_conv_min_tanh_forward, "Fused Conv + Min + Tanh forward pass");
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
    # Extract dimensions
    N, C_in, H, W = x.shape
    C_out, _, K, _ = conv_weight.shape
    
    # Calculate output dimensions
    OH = (H + 2 * conv_padding - K) // conv_stride + 1
    OW = (W + 2 * conv_padding - K) // conv_stride + 1
    
    # Create output tensor
    output = torch.empty(N, 1, OH, OW, device=x.device, dtype=x.dtype)
    
    # Call our custom fused kernel
    fused_ext.fused_conv_min_tanh_forward(
        x, conv_weight, conv_bias, output,
        N, C_in, C_out, H, W, K,
        conv_stride, conv_padding
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
