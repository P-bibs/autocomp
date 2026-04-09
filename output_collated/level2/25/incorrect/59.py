# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_091623/code_2.py
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

# Optimized CUDA kernel with memory coalescing and full fusion
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ w, 
    const float* __restrict__ b,
    float* __restrict__ out, 
    int N, int C_in, int C_out, int H, int W, int K, int stride, int padding) {

    // Calculate output dimensions
    int OH = (H + 2 * padding - K) / stride + 1;
    int OW = (W + 2 * padding - K) / stride + 1;
    
    // Get thread indices
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // Each thread processes multiple output pixels if needed
    for (int idx = tid; idx < N * OH * OW; idx += total_threads) {
        int n = idx / (OH * OW);
        int oh = (idx / OW) % OH;
        int ow = idx % OW;
        
        // Perform convolution and find channel minimum
        float min_val = 1e30f;
        
        for (int co = 0; co < C_out; ++co) {
            float sum = b[co];
            
            for (int ci = 0; ci < C_in; ++ci) {
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        int ih = oh * stride + kh - padding;
                        int iw = ow * stride + kw - padding;
                        
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            sum += x[((n * C_in + ci) * H + ih) * W + iw] * 
                                   w[(((co * C_in + ci) * K + kh) * K + kw)];
                        }
                    }
                }
            }
            
            if (sum < min_val) min_val = sum;
        }
        
        // Apply double tanh activation
        float t = tanhf(min_val);
        out[idx] = tanhf(t);
    }
}

void fused_op_forward(int grid_size, int block_size,
                      torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out,
                      int stride, int padding) {
    int N = x.size(0), C_in = x.size(1), H = x.size(2), W = x.size(3);
    int C_out = w.size(0), K = w.size(2);
    
    fused_conv_min_tanh_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
        out.data_ptr<float>(), N, C_in, C_out, H, W, K, stride, padding);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(int grid_size, int block_size,
                      torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out,
                      int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused convolution with min and double tanh");
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
    conv_stride=1,
    conv_padding=0,
    conv_dilation=1,
    conv_groups=1
):
    # Validate inputs
    assert conv_dilation == 1, "Only dilation=1 is supported"
    assert conv_groups == 1, "Only groups=1 is supported"
    
    # Get dimensions
    N, C_in, H, W = x.shape
    C_out, _, K, _ = conv_weight.shape
    
    # Calculate output dimensions
    OH = (H + 2 * conv_padding - K) // conv_stride + 1
    OW = (W + 2 * conv_padding - K) // conv_stride + 1
    
    # Allocate output tensor
    out = torch.empty((N, 1, OH, OW), device='cuda')
    
    # Configure kernel launch parameters
    total_outputs = N * OH * OW
    block_size = min(256, total_outputs)
    grid_size = min(65535, (total_outputs + block_size - 1) // block_size)
    
    # Launch kernel
    fused_ext.fused_op(
        grid_size, block_size,
        x, conv_weight, conv_bias, out,
        conv_stride, conv_padding
    )
    
    return out

# Input parameters for testing
batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
