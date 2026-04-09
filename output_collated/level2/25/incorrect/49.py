# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_090933/code_4.py
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
# CUDA Kernel: Fused Conv2D + Channel-Min + Tanh + Tanh
# ============================================================================

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Fast tanh approximation (more register-efficient)
__device__ __forceinline__ float fast_tanh(float x) {
    return tanh(x);
}

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ input,      // [N, C_in, H, W]
    const float* __restrict__ weight,     // [C_out, C_in, K, K]
    const float* __restrict__ bias,       // [C_out]
    float* __restrict__ output,           // [N, 1, OH, OW]
    int N, int C_in, int C_out, int H, int W, int K,
    int stride, int padding, int OH, int OW)
{
    // Each thread block handles one spatial output position across all batches
    int oh = blockIdx.y;
    int ow = blockIdx.x;
    int n = blockIdx.z;
    
    if (oh >= OH || ow >= OW || n >= N) return;
    
    // Shared memory for accumulating minimum value across channels
    __shared__ float shared_min_val;
    
    int tid = threadIdx.x;
    
    // Initialize min value
    if (tid == 0) {
        shared_min_val = INFINITY;
    }
    __syncthreads();
    
    // Each thread processes a subset of output channels
    for (int co = tid; co < C_out; co += blockDim.x) {
        float conv_result = bias[co];
        
        // Convolution computation
        for (int ci = 0; ci < C_in; ++ci) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    // Input spatial coordinates
                    int ih = oh * stride + kh - padding;
                    int iw = ow * stride + kw - padding;
                    
                    // Boundary check
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        int input_idx = ((n * C_in + ci) * H + ih) * W + iw;
                        int weight_idx = (((co * C_in + ci) * K) + kh) * K + kw;
                        
                        conv_result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Atomically update the minimum value
        atomicMinFloat(&shared_min_val, conv_result);
    }
    
    __syncthreads();
    
    // Apply double tanh activation and write result
    if (tid == 0) {
        float min_val = shared_min_val;
        float tanh1 = fast_tanh(min_val);
        float tanh2 = fast_tanh(tanh1);
        
        int out_idx = ((n * 1 + 0) * OH + oh) * OW + ow;
        output[out_idx] = tanh2;
    }
}

// Custom atomicMin for floats (CUDA doesn't have native atomicMin for floats)
__device__ void atomicMinFloat(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

// Host wrapper function
void fused_conv_min_tanh_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride, int padding, int OH, int OW)
{
    int N = input.size(0);
    int C_in = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int C_out = weight.size(0);
    int K = weight.size(2);
    
    // Grid and block dimensions
    dim3 gridDim(OW, OH, N);
    dim3 blockDim(min(C_out, 256));  // Limit threads per block to 256
    
    fused_conv_min_tanh_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out, H, W, K, stride, padding, OH, OW);
    
    C10_CUDA_CHECK(cudaGetLastError());
}
"""

# ============================================================================
# C++ Binding
# ============================================================================

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_min_tanh_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride, int padding, int OH, int OW);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_tanh", &fused_conv_min_tanh_forward, 
          "Fused Conv2D + ChannelMin + Tanh + Tanh kernel");
}
"""

# ============================================================================
# Compile CUDA Extension
# ============================================================================

try:
    fused_ext = load_inline(
        name='fused_conv_min_tanh',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        with_cuda=True,
        verbose=False
    )
except Exception as e:
    print(f"Compilation warning (non-fatal): {e}")
    fused_ext = None

# ============================================================================
# Python Interface
# ============================================================================

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
    Fused Conv2D + Channel-Min + Tanh + Tanh operation.
    
    Single kernel launches:
    - Conv2D with implicit GEMM
    - Channel-wise minimum reduction (across C_out)
    - Tanh activation applied twice
    """
    
    # Input validation
    assert conv_dilation == 1, "Only dilation=1 supported"
    assert conv_groups == 1, "Only groups=1 supported"
    assert x.is_cuda and x.dtype == torch.float32
    assert conv_weight.dtype == torch.float32
    
    # Extract dimensions
    N, C_in, H, W = x.shape
    C_out, _, K, _ = conv_weight.shape
    stride = conv_stride if isinstance(conv_stride, int) else conv_stride[0]
    padding = conv_padding if isinstance(conv_padding, int) else conv_padding[0]
    
    # Compute output dimensions
    OH = (H + 2 * padding - K) // stride + 1
    OW = (W + 2 * padding - K) // stride + 1
    
    # Output tensor: [N, 1, OH, OW] (reduced to 1 channel via min)
    output = torch.zeros(N, 1, OH, OW, dtype=torch.float32, device='cuda')
    
    # Call fused kernel
    if fused_ext is not None:
        fused_ext.fused_conv_min_tanh(
            x.contiguous(),
            conv_weight.contiguous(),
            conv_bias.contiguous(),
            output,
            stride, padding, OH, OW
        )
    else:
        # Fallback (should not reach here in production)
        raise RuntimeError("CUDA extension compilation failed")
    
    return output

# ============================================================================
# Test Parameters
# ============================================================================

batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
