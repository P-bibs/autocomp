# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_091623/code_10.py
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

# The optimized kernel uses thread-level parallelism for Channel-out (C_out) 
# and spatial-tiling for (OH, OW). 128 threads per block allow us to 
# saturate the RTX 2080Ti's warp scheduler while keeping footprint small enough 
# for efficient register allocation.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
    float* __restrict__ out, int N, int C_in, int C_out, int H, int W, int K, int OH, int OW) {

    // Map thread block to spatial output pixel (oh, ow) and batch index n
    int oh = blockIdx.y;
    int ow = blockIdx.x;
    int n = blockIdx.z;

    // Use threadIdx to parallelize over C_out for the reduction operation
    // Each thread calculates one partial component of the C_out reduction
    extern __shared__ float shared_vals[]; 
    int tid = threadIdx.x;

    if (tid < C_out) {
        float sum = b[tid];
        // Convolution: Standard sliding window
        for (int ci = 0; ci < C_in; ++ci) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    sum += x[((n * C_in + ci) * H + (oh + kh)) * W + (ow + kw)] * 
                           w[(((tid * C_in + ci) * K + kh) * K + kw)];
                }
            }
        }
        shared_vals[tid] = sum;
    }
    __syncthreads();

    // Perform reduction across shared memory (Min of C_out channels)
    if (tid == 0) {
        float min_val = shared_vals[0];
        for (int i = 1; i < C_out; ++i) {
            if (shared_vals[i] < min_val) min_val = shared_vals[i];
        }
        // Applying double tanh activation as requested
        float t = tanhf(min_val);
        out[(n * OH + oh) * OW + ow] = tanhf(t);
    }
}

void fused_op(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out) {
    int N = x.size(0), C_in = x.size(1), H = x.size(2), W = x.size(3);
    int C_out = w.size(0), K = w.size(2);
    int OH = H - K + 1, OW = W - K + 1;
    
    // Grid: Spatial dimensions and batch. Block: Thread-level C_out parallelism
    dim3 grid(OW, OH, N);
    dim3 block(C_out > 256 ? 256 : C_out); 
    
    fused_conv_min_tanh_kernel<<<grid, block, C_out * sizeof(float)>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), 
        out.data_ptr<float>(), N, C_in, C_out, H, W, K, OH, OW
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused Conv-Min-Tanh Operation");
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

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=0, 
                     conv_dilation=1, conv_groups=1):
    N, C_in, H, W = x.shape
    C_out, _, K, _ = conv_weight.shape
    OH, OW = H - K + 1, W - K + 1
    
    # Pre-allocate output tensor
    out = torch.empty((N, 1, OH, OW), device=x.device, dtype=x.dtype)
    
    # Execute fused custom kernel
    fused_ext.fused_op(x, conv_weight, conv_bias, out)
    
    return out
