# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_085904/code_2.py
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

#define TILE_HW 16
#define SHARED_SIZE (TILE_HW + 2)

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
    float* __restrict__ output, int N, int Ci, int Co, int H, int W, int K, 
    int stride, int padding, int OH, int OW) {
    
    extern __shared__ float shared_x[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int oh = blockIdx.y * TILE_HW + ty;
    int ow = blockIdx.x * TILE_HW + tx;
    int n = blockIdx.z;
    
    // Precompute bounds
    int IH = H + 2 * padding;
    int IW = W + 2 * padding;
    
    float min_val = INFINITY;
    
    // Load bias values into registers
    extern __shared__ float shared_bias[];
    if (ty == 0 && tx < 32 && tx < Co) {
        shared_bias[tx] = b[tx];
    }
    __syncthreads();
    
    // Iterate through output channels
    for (int co = 0; co < Co; ++co) {
        float sum = (co < 32) ? shared_bias[co] : b[co]; // Use shared bias when possible
        
        // Convolve with input feature map
        for (int ci = 0; ci < Ci; ++ci) {
            // Load tile to shared memory with halo region for padding
            for (int i = ty; i < SHARED_SIZE; i += TILE_HW) {
                for (int j = tx; j < SHARED_SIZE; j += TILE_HW) {
                    int ih = blockIdx.y * TILE_HW + i - padding;
                    int iw = blockIdx.x * TILE_HW + j - padding;
                    
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        shared_x[i * SHARED_SIZE + j] = x[((n * Ci + ci) * H + ih) * W + iw];
                    } else {
                        shared_x[i * SHARED_SIZE + j] = 0.0f;
                    }
                }
            }
            __syncthreads();
            
            // Perform convolution on the loaded tile
            if (oh < OH && ow < OW) {
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        int sh_i = ty + kh;
                        int sh_j = tx + kw;
                        if (sh_i < SHARED_SIZE && sh_j < SHARED_SIZE) {
                            sum += shared_x[sh_i * SHARED_SIZE + sh_j] *
                                   w[(((co * Ci + ci) * K + kh) * K + kw)];
                        }
                    }
                }
            }
            __syncthreads();
        }
        
        // Update minimum value across channels
        if (sum < min_val) {
            min_val = sum;
        }
    }
    
    // Apply double tanh activation and write result
    if (oh < OH && ow < OW) {
        float res = tanhf(tanhf(min_val));
        output[((n * 1) * OH + oh) * OW + ow] = res;
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out,
                      int stride, int padding) {
    int N = x.size(0); 
    int Ci = x.size(1); 
    int H = x.size(2); 
    int W = x.size(3);
    int Co = w.size(0); 
    int K = w.size(2);
    
    int OH = (H + 2 * padding - K) / stride + 1;
    int OW = (W + 2 * padding - K) / stride + 1;
    
    dim3 block(TILE_HW, TILE_HW);
    dim3 grid((OW + TILE_HW - 1) / TILE_HW, (OH + TILE_HW - 1) / TILE_HW, N);
    
    size_t shared_mem_size = SHARED_SIZE * SHARED_SIZE * sizeof(float) + 32 * sizeof(float);
    
    fused_conv_min_tanh_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
        out.data_ptr<float>(), N, Ci, Co, H, W, K, stride, padding, OH, OW);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out,
                      int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv-Min-Tanh");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=compute_75', '-code=sm_75'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1):
    assert conv_dilation == 1, "Only dilation=1 supported"
    assert conv_groups == 1, "Only groups=1 supported"
    
    N, Ci, H, W = x.shape
    Co, _, K, _ = conv_weight.shape
    
    OH = (H + 2 * conv_padding - K) // conv_stride + 1
    OW = (W + 2 * conv_padding - K) // conv_stride + 1
    
    out = torch.empty((N, 1, OH, OW), device='cuda')
    fused_ext.fused_op(x, conv_weight, conv_bias, out, conv_stride, conv_padding)
    return out
