# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_083856/code_2.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 16

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out, int H, int W, int K) {

    int OH = H - K + 1;
    int OW = W - K + 1;
    
    // Shared memory for tile of input and weights
    extern __shared__ float shared_mem[];
    float* s_x = shared_mem;
    float* s_w = &shared_mem[TILE_SIZE * TILE_SIZE * K * K];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Block indices
    int block_co = blockIdx.x;  // Output channel block
    int block_oh = blockIdx.y;  // Output height block
    int n_ow = blockIdx.z;      // Batch and output width combined
    
    int n = n_ow / OW;
    int ow = n_ow % OW;
    
    // Thread indices within tile
    int co_local = ty;
    int oh_local = tx;
    
    // Global indices
    int co = block_co * TILE_SIZE + co_local;
    int oh = block_oh * TILE_SIZE + oh_local;
    
    if (oh >= OH) return;
    
    float sum = 0.0f;
    
    // Loop over input channels in tiles
    for (int ci_block = 0; ci_block < (C_in + TILE_SIZE - 1) / TILE_SIZE; ++ci_block) {
        // Load input tile to shared memory
        for (int i = ty, idx = 0; i < K && idx < TILE_SIZE*K; i += TILE_SIZE) {
            for (int j = tx; j < K && (i*K+j) < TILE_SIZE*K; j += TILE_SIZE) {
                int ci = ci_block * TILE_SIZE + idx;
                if (ci < C_in) {
                    s_x[idx * K * K + i * K + j] = x[((n * C_in + ci) * H + (oh + i)) * W + (ow + j)];
                } else {
                    s_x[idx * K * K + i * K + j] = 0.0f;
                }
                idx++;
            }
        }
        
        // Load weights to shared memory
        for (int i = ty; i < K; i += TILE_SIZE) {
            for (int j = tx; j < K; j += TILE_SIZE) {
                int ci = ci_block * TILE_SIZE + (i*K+j) % TILE_SIZE;
                if (co < C_out && ci < C_in) {
                    s_w[(i*K+j) * K * K] = weight[((co * C_in + ci) * K + i) * K + j];
                } else {
                    s_w[(i*K+j) * K * K] = 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        // Compute partial convolution
        if (co < C_out) {
            for (int ci_idx = 0; ci_idx < TILE_SIZE && (ci_block * TILE_SIZE + ci_idx) < C_in; ++ci_idx) {
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        sum += s_x[ci_idx * K * K + kh * K + kw] * 
                               s_w[ci_idx * K * K + kh * K + kw];
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Add bias
    if (co < C_out) {
        sum += bias[co];
        
        // Apply double tanh
        float result = tanhf(tanhf(sum));
        
        // Store result
        output[((n * C_out + co) * OH + oh) * OW + ow] = result;
    }
}

void fused_conv_min_tanh_launcher(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output) {
    
    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int C_out = weight.size(0);
    int K = weight.size(2);
    
    int OH = H - K + 1;
    int OW = W - K + 1;
    
    // Grid dimensions
    dim3 grid(
        (C_out + TILE_SIZE - 1) / TILE_SIZE,  // CO blocks
        (OH + TILE_SIZE - 1) / TILE_SIZE,     // OH blocks
        N * OW                                // N * OW threads
    );
    
    // Block dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);
    
    // Shared memory size
    int shared_mem_size = 2 * TILE_SIZE * K * K * sizeof(float);
    
    fused_conv_min_tanh_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out, H, W, K
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_min_tanh_launcher(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &fused_conv_min_tanh_launcher, "Fused Conv + Min + Tanh + Tanh");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_ext',
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
    conv_stride=1,
    conv_padding=0,
    conv_dilation=1,
    conv_groups=1
):
    # Only support stride 1, padding 0 for the optimized implicit GEMM path
    assert conv_stride == 1 and conv_padding == 0 and conv_dilation == 1 and conv_groups == 1
    
    N, C_in, H, W = x.shape
    C_out, _, K, _ = conv_weight.shape
    OH, OW = H - K + 1, W - K + 1
    
    # Allocate output tensor
    output = torch.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)
    
    # Launch fused kernel
    fused_ext.fused_conv(x, conv_weight, conv_bias, output)
    
    # Min reduction across channel dimension as required
    res = torch.min(output, dim=1, keepdim=True)[0]
    
    return res
