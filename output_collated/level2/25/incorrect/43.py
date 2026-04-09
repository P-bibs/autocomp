# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_085904/code_4.py
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

# ============================================================================
# CUDA Kernel for Fused Conv2D + ChannelMin + DoubleTanh
# ============================================================================

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x,           // [N, C_in, H, W]
    const float* __restrict__ weight,      // [C_out, C_in, K, K]
    const float* __restrict__ bias,        // [C_out]
    float* __restrict__ output,            // [N, 1, OH, OW]
    int N, int C_in, int C_out, int H, int W, int K,
    int stride, int padding) {

    // Shared memory for input patch
    extern __shared__ float smem_input[];
    
    // Block and thread indices
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tid = tidy * blockDim.x + tidx;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int bidz = blockIdx.z;
    
    // Output dimensions
    int OH = (H + 2 * padding - K) / stride + 1;
    int OW = (W + 2 * padding - K) / stride + 1;
    
    // Starting output position for this block
    int oh = bidy;
    int ow = bidx;
    int n = bidz;
    
    if (oh >= OH || ow >= OW) return;
    
    // Thread-local accumulator for each output channel
    float accum[64];  // Assuming max 64 output channels for register efficiency
    float min_val = 1e30f;
    
    // Initialize accumulators with bias
    #pragma unroll 4
    for (int co = 0; co < C_out && co < 64; ++co) {
        accum[co] = bias[co];
    }
    
    // Convolution computation
    for (int ci = 0; ci < C_in; ++ci) {
        // Load input patch into shared memory
        int patch_h = K;
        int patch_w = K;
        int patch_size = patch_h * patch_w;
        
        for (int idx = tid; idx < patch_size; idx += blockDim.x * blockDim.y) {
            int kh = idx / patch_w;
            int kw = idx % patch_w;
            int ih = oh * stride + kh - padding;
            int iw = ow * stride + kw - padding;
            
            float val = 0.0f;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                val = x[((n * C_in + ci) * H + ih) * W + iw];
            }
            smem_input[idx] = val;
        }
        __syncthreads();
        
        // Perform convolution for this input channel
        #pragma unroll 3
        for (int co = 0; co < C_out && co < 64; ++co) {
            float sum = 0.0f;
            #pragma unroll 3
            for (int kh = 0; kh < K; ++kh) {
                #pragma unroll 3
                for (int kw = 0; kw < K; ++kw) {
                    float in_val = smem_input[kh * K + kw];
                    float w_val = weight[(((co * C_in + ci) * K + kh) * K + kw)];
                    sum += in_val * w_val;
                }
            }
            accum[co] += sum;
        }
        __syncthreads();
    }
    
    // Reduction phase: find minimum across channels
    #pragma unroll 4
    for (int co = 0; co < C_out && co < 64; ++co) {
        min_val = fminf(min_val, accum[co]);
    }
    
    // Apply double tanh: tanh(tanh(x))
    float tanh1 = tanhf(min_val);
    float tanh2 = tanhf(tanh1);
    
    // Store result
    output[((n * 1 + 0) * OH + oh) * OW + ow] = tanh2;
}

void fused_conv_min_tanh_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding) {
    
    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int C_out = weight.size(0);
    int K = weight.size(2);
    
    int OH = (H + 2 * padding - K) / stride + 1;
    int OW = (W + 2 * padding - K) / stride + 1;
    
    // Shared memory size for input patch
    int shared_mem_size = K * K * sizeof(float);
    
    // Grid and block dimensions
    dim3 blocks(OW, OH, N);
    dim3 threads(16, 16);
    
    fused_conv_min_tanh_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out, H, W, K,
        stride, padding);
    
    cudaDeviceSynchronize();
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_min_tanh_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv_min_tanh_forward, "Fused Conv2D + Min + DoubleTanh");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_min_tanh',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
    with_cuda=True,
    verbose=False
)

# ============================================================================
# PyTorch Interface
# ============================================================================

# Input parameters
batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3
conv_stride = 1
conv_padding = 1
conv_dilation = 1
conv_groups = 1

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]

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
    """
    Fused convolution + channel-min + double-tanh operation.
    
    Replaces:
        x = torch.conv2d(x, conv_weight, conv_bias, stride=conv_stride, padding=conv_padding, ...)
        x = torch.tanh(torch.min(x, dim=1, keepdim=True)[0])
        x = torch.tanh(x)
    
    With a single custom CUDA kernel that performs all operations in one pass.
    """
    
    N = x.size(0)
    C_in = x.size(1)
    H = x.size(2)
    W = x.size(3)
    C_out = conv_weight.size(0)
    K = conv_weight.size(2)
    
    OH = (H + 2 * conv_padding - K) // conv_stride + 1
    OW = (W + 2 * conv_padding - K) // conv_stride + 1
    
    # Allocate output tensor
    output = torch.empty((N, 1, OH, OW), dtype=x.dtype, device=x.device)
    
    # Call fused kernel
    fused_ext.fused_op(x, conv_weight, conv_bias, output, conv_stride, conv_padding)
    
    return output
