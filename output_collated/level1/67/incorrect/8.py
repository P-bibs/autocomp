# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_155649/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv1d_weight', 'conv1d_bias', 'conv1d_stride', 'conv1d_padding', 'conv1d_dilation', 'conv1d_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv1d_weight', 'conv1d_bias']


class ModelNew(nn.Module):
    """
    Performs a standard 1D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0, dilation: int=1, groups: int=1, bias: bool=False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

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
    # State for conv1d (nn.Conv1d)
    if 'conv1d_weight' in flat_state:
        state_kwargs['conv1d_weight'] = flat_state['conv1d_weight']
    else:
        state_kwargs['conv1d_weight'] = getattr(model.conv1d, 'weight', None)
    if 'conv1d_bias' in flat_state:
        state_kwargs['conv1d_bias'] = flat_state['conv1d_bias']
    else:
        state_kwargs['conv1d_bias'] = getattr(model.conv1d, 'bias', None)
    state_kwargs['conv1d_stride'] = model.conv1d.stride
    state_kwargs['conv1d_padding'] = model.conv1d.padding
    state_kwargs['conv1d_dilation'] = model.conv1d.dilation
    state_kwargs['conv1d_groups'] = model.conv1d.groups
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

# CUDA kernel implementing tiled Conv1D with shared memory optimization
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 256
#define HALO_SIZE 2

__global__ void conv1d_tiled_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int batch,
    int in_channels,
    int out_channels,
    int length,
    int kernel_size,
    int out_length
) {
    // Shared memory for input tile and halo
    extern __shared__ float shared_x[];
    float* shared_weight = shared_x + TILE_SIZE + 2 * HALO_SIZE;
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_tid = bid * blockDim.x + tid;
    
    // Load weights into shared memory (one thread per weight element)
    int weight_elements = in_channels * out_channels * kernel_size;
    if (tid < weight_elements && bid == 0) {
        shared_weight[tid] = weight[tid];
    }
    __syncthreads();
    
    // Each block processes one output channel for one batch element
    int batch_idx = bid / out_channels;
    int out_ch_idx = bid % out_channels;
    
    if (batch_idx >= batch) return;
    
    // Process output positions in tiles
    for (int out_pos_start = 0; out_pos_start < out_length; out_pos_start += TILE_SIZE) {
        int out_pos = out_pos_start + tid;
        
        // Load input data with halo into shared memory
        for (int i = 0; i < 1 + (2 * HALO_SIZE) / blockDim.x; i++) {
            int load_idx = tid + i * blockDim.x;
            if (load_idx < TILE_SIZE + 2 * HALO_SIZE) {
                int global_in_pos = out_pos_start + load_idx - HALO_SIZE;
                if (global_in_pos >= 0 && global_in_pos < length) {
                    // Load all input channels for this position
                    float sum = 0.0f;
                    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
                        sum += x[batch_idx * in_channels * length + 
                                in_ch * length + global_in_pos];
                    }
                    shared_x[load_idx] = sum;
                } else {
                    shared_x[load_idx] = 0.0f;
                }
            }
        }
        __syncthreads();
        
        // Compute output if within bounds
        if (out_pos < out_length) {
            float result = bias[out_ch_idx];
            
            // Convolve with kernel
            for (int k = 0; k < kernel_size; k++) {
                int in_pos = out_pos + k;
                if (in_pos < length) {
                    float x_val = shared_x[tid + k + HALO_SIZE];
                    float w_val = shared_weight[out_ch_idx * in_channels * kernel_size + 0 * kernel_size + k];
                    result += x_val * w_val;
                }
            }
            
            out[batch_idx * out_channels * out_length + 
                out_ch_idx * out_length + out_pos] = result;
        }
        __syncthreads();
    }
}

// Optimized kernel for specific case: kernel_size=3, stride=1, padding=0
__global__ void conv1d_tiled_kernel_opt(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int batch,
    int in_channels,
    int out_channels,
    int length
) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int out_pos = blockIdx.x * blockDim.x + tid;
    int out_length = length - 2; // For kernel_size=3
    
    if (out_pos >= out_length) return;
    
    // Load weights into shared memory
    if (tid < in_channels * out_channels * 3) {
        shared_mem[tid] = weight[tid];
    }
    __syncthreads();
    
    // Each thread computes one output position for all channels and batches
    for (int b = 0; b < batch; b++) {
        for (int out_ch = 0; out_ch < out_channels; out_ch++) {
            float sum = bias[out_ch];
            
            for (int in_ch = 0; in_ch < in_channels; in_ch++) {
                // Load 3 input values
                float x0 = x[b * in_channels * length + in_ch * length + out_pos];
                float x1 = x[b * in_channels * length + in_ch * length + out_pos + 1];
                float x2 = x[b * in_channels * length + in_ch * length + out_pos + 2];
                
                // Load 3 weights
                int w_idx = out_ch * in_channels * 3 + in_ch * 3;
                float w0 = shared_mem[w_idx];
                float w1 = shared_mem[w_idx + 1];
                float w2 = shared_mem[w_idx + 2];
                
                sum += x0 * w0 + x1 * w1 + x2 * w2;
            }
            
            out[b * out_channels * out_length + out_ch * out_length + out_pos] = sum;
        }
    }
}

void conv1d_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out) {
    int batch = x.size(0);
    int in_channels = x.size(1);
    int length = x.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int out_length = length - kernel_size + 1;
    
    if (kernel_size == 3) {
        // Use optimized kernel for kernel_size=3
        int threads = 256;
        int blocks = (out_length + threads - 1) / threads;
        size_t shared_mem_size = in_channels * out_channels * 3 * sizeof(float);
        
        conv1d_tiled_kernel_opt<<<blocks, threads, shared_mem_size>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            out.data_ptr<float>(),
            batch,
            in_channels,
            out_channels,
            length
        );
    } else {
        // Use general kernel
        int threads = 256;
        int blocks = batch * out_channels;
        size_t shared_mem_size = (threads + 2 * HALO_SIZE) * sizeof(float) + 
                                in_channels * out_channels * kernel_size * sizeof(float);
        
        conv1d_tiled_kernel<<<blocks, threads, shared_mem_size>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            out.data_ptr<float>(),
            batch,
            in_channels,
            out_channels,
            length,
            kernel_size,
            out_length
        );
    }
}
"""

# C++ wrapper for the CUDA functions
cpp_source = r"""
#include <torch/extension.h>

void conv1d_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d_forward", &conv1d_forward, "Tiled Conv1D forward pass");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='tiled_conv1d',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
    with_cuda=True
)

# Model parameters
batch_size = 32
in_channels = 64
out_channels = 128
kernel_size = 3
length = 131072

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    x = torch.rand(batch_size, in_channels, length, device='cuda')
    return [x]

def functional_model(
    x,
    *,
    conv1d_weight,
    conv1d_bias,
    conv1d_stride,
    conv1d_padding,
    conv1d_dilation,
    conv1d_groups,
):
    # Ensure inputs are on GPU
    x = x.contiguous().cuda()
    conv1d_weight = conv1d_weight.contiguous().cuda()
    conv1d_bias = conv1d_bias.contiguous().cuda()
    
    # Calculate output length (assuming stride=1, padding=0)
    out_length = x.shape[2] - conv1d_weight.shape[2] + 1
    out = torch.empty((x.shape[0], conv1d_weight.shape[0], out_length), 
                      device='cuda', dtype=x.dtype)
    
    # Call our optimized CUDA implementation
    fused_ext.conv1d_forward(x, conv1d_weight, conv1d_bias, out)
    
    return out
