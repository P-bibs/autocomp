# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150347/code_6.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'subtract_value_1', 'subtract_value_2']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'subtract_value_1', 'subtract_value_2']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, subtracts two values, applies Mish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

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
    if 'subtract_value_1' in flat_state:
        state_kwargs['subtract_value_1'] = flat_state['subtract_value_1']
    else:
        state_kwargs['subtract_value_1'] = getattr(model, 'subtract_value_1')
    if 'subtract_value_2' in flat_state:
        state_kwargs['subtract_value_2'] = flat_state['subtract_value_2']
    else:
        state_kwargs['subtract_value_2'] = getattr(model, 'subtract_value_2')
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

# Define the optimized CUDA kernel using Implicit GEMM approach
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float mish(float x) {
    return x * tanhf(log1pf(expf(x)));
}

__global__ void fused_conv_mish_implicit_gemm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int B,
    const int C_in,
    const int H,
    const int W,
    const int C_out,
    const int k,
    const float sub1,
    const float sub2
) {
    const int H_out = H - k + 1;
    const int W_out = W - k + 1;

    // Thread and block indices
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    // Each block processes multiple output pixels per channel
    const int TILE_X = 16;
    const int TILE_Y = 16;

    // Shared memory for input tile and partial accumulations
    extern __shared__ float shared_mem[];
    float* sh_input = shared_mem;
    float* sh_weight = &shared_mem[TILE_X * TILE_Y * C_in];

    // Per-thread registers for accumulation
    float thread_result = 0.0f;

    // Load input tile into shared memory cooperatively
    for (int ic = 0; ic < C_in; ++ic) {
        for (int i = tid_y; i < TILE_Y; i += blockDim.y) {
            for (int j = tid_x; j < TILE_X; j += blockDim.x) {
                int out_y = block_y * TILE_Y + i;
                int out_x = block_x * TILE_X + j;
                if (out_y < H_out && out_x < W_out) {
                    // Compute base address in input tensor
                    int idx_base = ((block_z / C_out) * C_in + ic) * H * W;
                    float val = 0.0f;
                    for (int ki = 0; ki < k; ++ki) {
                        for (int kj = 0; kj < k; ++kj) {
                            val += input[idx_base + (out_y + ki) * W + (out_x + kj)] *
                                   weight[(block_z % C_out) * C_in * k * k + ic * k * k + ki * k + kj];
                        }
                    }
                    sh_input[i * TILE_X * C_in + j * C_in + ic] = val;
                }
            }
        }
    }

    __syncthreads();

    // Accumulate results
    for (int i = tid_y; i < TILE_Y; i += blockDim.y) {
        for (int j = tid_x; j < TILE_X; j += blockDim.x) {
            int out_y = block_y * TILE_Y + i;
            int out_x = block_x * TILE_X + j;
            if (out_y < H_out && out_x < W_out) {
                thread_result = bias[block_z % C_out];
                for (int ic = 0; ic < C_in; ++ic) {
                    thread_result += sh_input[i * TILE_X * C_in + j * C_in + ic];
                }
                float activated = mish(thread_result - sub1 - sub2);
                int out_idx = ((block_z / C_out) * C_out + (block_z % C_out)) * H_out * W_out + out_y * W_out + out_x;
                output[out_idx] = activated;
            }
        }
    }
}

void fused_conv_mish_implicit_gemm(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float s1,
    float s2
) {
    const int B = input.size(0);
    const int C_in = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int C_out = weight.size(0);
    const int k = weight.size(2);
    const int H_out = H - k + 1;
    const int W_out = W - k + 1;

    const dim3 block(16, 16);
    const dim3 grid((W_out + 15) / 16, (H_out + 15) / 16, B * C_out);

    // Shared memory size: input tile + weight tile
    const size_t shmem_input = 16 * 16 * C_in * sizeof(float);
    const size_t shmem_weight = 0; // We're not loading weights into shared mem in this version
    const size_t shmem_bytes = shmem_input + shmem_weight;

    fused_conv_mish_implicit_gemm_kernel<<<grid, block, shmem_bytes>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C_in, H, W, C_out, k, s1, s2
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_mish_implicit_gemm(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                                   torch::Tensor output, float s1, float s2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &fused_conv_mish_implicit_gemm, "Fused Conv + Mish using Implicit GEMM");
}
"""

# Load the extension
fused_ext = load_inline(
    name='fused_conv_mish_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=0, 
                     conv_dilation=1, conv_groups=1, subtract_value_1, subtract_value_2):
    # Ensure tensors are contiguous
    x = x.contiguous()
    conv_weight = conv_weight.contiguous()
    conv_bias = conv_bias.contiguous()
    
    # Extract dimensions
    batch, _, h, w = x.shape
    k = conv_weight.shape[2]
    out_h = h - k + 1
    out_w = w - k + 1
    out_c = conv_weight.size(0)
    
    # Create output tensor
    out = torch.empty((batch, out_c, out_h, out_w), device=x.device, dtype=x.dtype)
    
    # Launch fused kernel
    fused_ext.fused_conv(x, conv_weight, conv_bias, out, 
                         float(subtract_value_1), float(subtract_value_2))
    
    return out
