# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_142828/code_9.py
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

# Optimized CUDA kernel with shared memory tiling
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_DIM 16

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int in_c, int in_h, int in_w,
    int out_c, int k, int out_h, int out_w,
    float sub1, float sub2) {
    
    // Shared memory for input tile: [in_c][(TILE_DIM + k - 1) * (TILE_DIM + k - 1)]
    extern __shared__ float shmem[];
    
    // Decode block indices
    int oc = blockIdx.z % out_c;
    int b = blockIdx.z / out_c;
    
    // Calculate the origin of the current tile in the output
    int tile_origin_h = blockIdx.y * TILE_DIM;
    int tile_origin_w = blockIdx.x * TILE_DIM;
    
    // Thread coordinates within the tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Load input tile into shared memory
    for (int ic = 0; ic < in_c; ++ic) {
        int shm_base = ic * (TILE_DIM + k - 1) * (TILE_DIM + k - 1);
        
        // Each thread loads multiple elements if necessary
        for (int i = ty; i < TILE_DIM + k - 1; i += TILE_DIM) {
            for (int j = tx; j < TILE_DIM + k - 1; j += TILE_DIM) {
                int in_row = tile_origin_h + i;
                int in_col = tile_origin_w + j;
                
                float val = 0.0f;
                if (in_row >= 0 && in_row < in_h && in_col >= 0 && in_col < in_w) {
                    val = input[((b * in_c + ic) * in_h + in_row) * in_w + in_col];
                }
                
                shmem[shm_base + i * (TILE_DIM + k - 1) + j] = val;
            }
        }
    }
    
    __syncthreads();
    
    // Calculate output position for this thread
    int out_y = tile_origin_h + ty;
    int out_x = tile_origin_w + tx;
    
    if (out_y < out_h && out_x < out_w) {
        // Compute convolution + bias
        float acc = bias[oc];
        
        // Weight pointer for this output channel
        const float* weight_ptr = weight + oc * k * k * in_c;
        
        for (int ic = 0; ic < in_c; ++ic) {
            int shm_base = ic * (TILE_DIM + k - 1) * (TILE_DIM + k - 1);
            
            for (int i = 0; i < k; ++i) {
                for (int j = 0; j < k; ++j) {
                    float input_val = shmem[shm_base + (ty + i) * (TILE_DIM + k - 1) + (tx + j)];
                    float weight_val = weight_ptr[(i * k + j) * in_c + ic];
                    acc += input_val * weight_val;
                }
            }
        }
        
        // Apply subtractions and Mish activation
        float val = acc - sub1 - sub2;
        output[((b * out_c + oc) * out_h + out_y) * out_w + out_x] = val * tanhf(logf(1.0f + expf(val)));
    }
}

void fused_conv_mish(int blocks_x, int blocks_y, int blocks_z,
                     int threads_x, int threads_y,
                     const float* input, const float* weight,
                     const float* bias, float* output,
                     int batch, int in_c, int in_h, int in_w,
                     int out_c, int k, int out_h, int out_w,
                     float sub1, float sub2) {
    dim3 blocks(blocks_x, blocks_y, blocks_z);
    dim3 threads(threads_x, threads_y);
    size_t shm_bytes = in_c * (TILE_DIM + k - 1) * (TILE_DIM + k - 1) * sizeof(float);
    
    fused_conv_mish_kernel<<<blocks, threads, shm_bytes>>>(
        input, weight, bias, output,
        batch, in_c, in_h, in_w,
        out_c, k, out_h, out_w,
        sub1, sub2);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_mish(int blocks_x, int blocks_y, int blocks_z,
                     int threads_x, int threads_y,
                     const float* input, const float* weight,
                     const float* bias, float* output,
                     int batch, int in_c, int in_h, int in_w,
                     int out_c, int k, int out_h, int out_w,
                     float sub1, float sub2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &fused_conv_mish, "Fused Convolution + Mish with shared memory tiling");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias,
                     subtract_value_1, subtract_value_2):
    # Prepare weights for coalesced access: [out_c, k, k, in_c]
    w_reordered = conv_weight.permute(0, 2, 3, 1).contiguous()
    
    B, _, H, W = x.shape
    K = conv_weight.shape[2]
    out_h = H - K + 1
    out_w = W - K + 1
    out = torch.empty((B, conv_weight.size(0), out_h, out_w), device=x.device, dtype=x.dtype)
    
    # Grid / block calculation
    TILE_DIM = 16
    blocks_x = (out_w + TILE_DIM - 1) // TILE_DIM
    blocks_y = (out_h + TILE_DIM - 1) // TILE_DIM
    blocks_z = B * conv_weight.size(0)  # batch * out_channels
    threads_x = TILE_DIM
    threads_y = TILE_DIM
    
    fused_ext.fused_conv(blocks_x, blocks_y, blocks_z,
                         threads_x, threads_y,
                         x, w_reordered, conv_bias, out,
                         B, x.shape[1], H, W,
                         conv_weight.size(0), K, out_h, out_w,
                         subtract_value_1, subtract_value_2)
    return out
