# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160431/code_1.py
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

# Optimization: Using Implicit GEMM with Shared Memory Tiling
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void conv1d_imgemm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int B, int C, int L, int OC, int K, int L_out
) {
    // Shared memory for input tile and weight tile
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + TILE_SIZE * TILE_SIZE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int batch_idx = blockIdx.x;
    int oc_tile_start = blockIdx.y * TILE_SIZE;
    int l_tile_start = blockIdx.z * TILE_SIZE;
    
    int oc_idx = oc_tile_start + ty;
    int l_idx = l_tile_start + tx;

    float sum = 0.0f;
    
    // Loop over input channels in tiles
    for (int c_tile = 0; c_tile < C; c_tile += TILE_SIZE) {
        // Load input tile into shared memory
        for (int i = 0; i < TILE_SIZE; i += blockDim.y) {
            for (int j = 0; j < TILE_SIZE; j += blockDim.x) {
                int c = c_tile + i + ty;
                int pos = l_idx + j;
                if (c < C && pos < L) {
                    shared_input[(i + ty) * TILE_SIZE + (j + tx)] = 
                        input[batch_idx * C * L + c * L + pos];
                } else {
                    shared_input[(i + ty) * TILE_SIZE + (j + tx)] = 0.0f;
                }
            }
        }

        // Load weight tile into shared memory
        for (int i = 0; i < TILE_SIZE; i += blockDim.y) {
            for (int j = 0; j < TILE_SIZE; j += blockDim.x) {
                int oc = oc_tile_start + i + ty;
                int c = c_tile + j + tx;
                if (oc < OC && c < C) {
                    shared_weight[(i + ty) * TILE_SIZE + (j + tx)] = 
                        weight[oc * C * K + c * K + 0]; // Simplified for kernel size 3 with unrolling
                } else {
                    shared_weight[(i + ty) * TILE_SIZE + (j + tx)] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute partial sum
        for (int k = 0; k < K; k++) {
            for (int c_inner = 0; c_inner < TILE_SIZE; c_inner++) {
                int global_c = c_tile + c_inner;
                if (global_c < C && l_idx < L_out && oc_idx < OC) {
                    float input_val = 0.0f;
                    if (l_idx + k < L) {
                        input_val = input[batch_idx * C * L + global_c * L + (l_idx + k)];
                    }
                    float weight_val = 0.0f;
                    if (oc_idx < OC && global_c < C) {
                        weight_val = weight[oc_idx * C * K + global_c * K + k];
                    }
                    sum += input_val * weight_val;
                }
            }
        }

        __syncthreads();
    }

    // Write output
    if (l_idx < L_out && oc_idx < OC) {
        output[batch_idx * OC * L_out + oc_idx * L_out + l_idx] = sum;
    }
}

// Optimized kernel for kernel_size=3
__global__ void conv1d_imgemm_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int B, int C, int L, int OC, int K, int L_out
) {
    int batch_idx = blockIdx.x;
    int oc_idx = blockIdx.y;
    int l_idx = threadIdx.x + blockIdx.z * blockDim.x;
    
    if (l_idx >= L_out || batch_idx >= B || oc_idx >= OC) return;
    
    float sum = 0.0f;
    
    // Unroll the kernel_size=3 loop
    for (int c = 0; c < C; c++) {
        sum += input[batch_idx * C * L + c * L + l_idx] * weight[oc_idx * C * K + c * K + 0];
        sum += input[batch_idx * C * L + c * L + l_idx + 1] * weight[oc_idx * C * K + c * K + 1];
        sum += input[batch_idx * C * L + c * L + l_idx + 2] * weight[oc_idx * C * K + c * K + 2];
    }
    
    output[batch_idx * OC * L_out + oc_idx * L_out + l_idx] = sum;
}

void launch_conv1d(torch::Tensor x, torch::Tensor weight, torch::Tensor output) {
    int B = x.size(0), C = x.size(1), L = x.size(2);
    int OC = weight.size(0), K = weight.size(2);
    int L_out = L - K + 1;
    
    // Use optimized kernel for kernel_size=3
    if (K == 3) {
        dim3 threads(256);
        dim3 blocks(B, OC, (L_out + threads.x - 1) / threads.x);
        
        conv1d_imgemm_optimized_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
            B, C, L, OC, K, L_out);
    } else {
        // Fallback to general kernel
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks(B, (OC + TILE_SIZE - 1) / TILE_SIZE, (L_out + TILE_SIZE - 1) / TILE_SIZE);
        int shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
        
        conv1d_imgemm_kernel<<<blocks, threads, shared_mem_size>>>(
            x.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
            B, C, L, OC, K, L_out);
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_conv1d(torch::Tensor x, torch::Tensor weight, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_conv1d", &launch_conv1d, "1D Convolution using Implicit GEMM");
}
"""

fused_ext = load_inline(
    name='fused_conv1d_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

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
    # Constraint: Support simple case (stride=1, padding=0, dilation=1, groups=1)
    assert conv1d_stride == 1
    assert conv1d_padding == 0
    assert conv1d_dilation == 1
    assert conv1d_groups == 1
    
    batch_size, in_channels, length = x.shape
    out_channels = conv1d_weight.shape[0]
    kernel_size = conv1d_weight.shape[2]
    out_length = length - kernel_size + 1
    
    output = torch.empty((batch_size, out_channels, out_length), device=x.device, dtype=x.dtype)
    fused_ext.launch_conv1d(x, conv1d_weight, output)
    
    if conv1d_bias is not None:
        output += conv1d_bias.view(1, -1, 1)
    return output

# Setup parameters
batch_size = 32
in_channels = 64
out_channels = 128
kernel_size = 3
length = 131072

conv1d_weight = torch.randn(out_channels, in_channels, kernel_size).cuda()
conv1d_bias = torch.randn(out_channels).cuda()

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    x = torch.rand(batch_size, in_channels, length).cuda()
    return [x]
