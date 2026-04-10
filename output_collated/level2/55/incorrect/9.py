# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_153614/code_10.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'kernel_size', 'scale_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['matmul_weight', 'matmul_bias', 'max_pool_kernel_size', 'max_pool_stride', 'max_pool_padding', 'max_pool_dilation', 'max_pool_ceil_mode', 'max_pool_return_indices', 'scale_factor']
REQUIRED_FLAT_STATE_NAMES = ['matmul_weight', 'matmul_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs matrix multiplication, max pooling, sum, and scaling.
    """

    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)
        self.scale_factor = scale_factor

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
    # State for matmul (nn.Linear)
    if 'matmul_weight' in flat_state:
        state_kwargs['matmul_weight'] = flat_state['matmul_weight']
    else:
        state_kwargs['matmul_weight'] = getattr(model.matmul, 'weight', None)
    if 'matmul_bias' in flat_state:
        state_kwargs['matmul_bias'] = flat_state['matmul_bias']
    else:
        state_kwargs['matmul_bias'] = getattr(model.matmul, 'bias', None)
    # State for max_pool (nn.MaxPool1d)
    state_kwargs['max_pool_kernel_size'] = model.max_pool.kernel_size
    state_kwargs['max_pool_stride'] = model.max_pool.stride
    state_kwargs['max_pool_padding'] = model.max_pool.padding
    state_kwargs['max_pool_dilation'] = model.max_pool.dilation
    state_kwargs['max_pool_ceil_mode'] = model.max_pool.ceil_mode
    state_kwargs['max_pool_return_indices'] = model.max_pool.return_indices
    if 'scale_factor' in flat_state:
        state_kwargs['scale_factor'] = flat_state['scale_factor']
    else:
        state_kwargs['scale_factor'] = getattr(model, 'scale_factor')
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

# CUDA implementation
# The kernel computes dot products in chunks. 
# To stay within memory limits, we avoid storing the intermediate 32k output vector.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_pool_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const int B, const int I, const int O, 
    const int K, const int S, const float scale) {
    
    // Each block processes one batch item b
    int b = blockIdx.x;
    int num_pools = (O - K) / S + 1;
    
    // Shared memory for a small staging buffer if needed, 
    // but here we focus on register-local accumulation for performance
    for (int p = threadIdx.x; p < num_pools; p += blockDim.x) {
        float window_max = -3.40282e+38f; // FLT_MIN equivalent
        
        for (int k = 0; k < K; ++k) {
            int out_idx = p * S + k;
            float val = bias ? bias[out_idx] : 0.0f;
            
            const float* x_ptr = x + b * I;
            const float* w_ptr = weight + out_idx * I;
            
            float dot = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < I; ++i) {
                dot += x_ptr[i] * w_ptr[i];
            }
            val += dot;
            if (val > window_max) window_max = val;
        }
        
        // Accumulate in a thread-local variable, then atomicAdd at the end
        // to reduce contention on the output memory
        atomicAdd(out + b, window_max * scale / (float)num_pools); 
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out, 
                      int B, int I, int O, int K, int S, float scale) {
    const int threads = 256;
    fused_linear_pool_kernel<<<B, threads>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), 
        out.data_ptr<float>(), B, I, O, K, S, scale);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out, 
                      int B, int I, int O, int K, int S, float scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Linear-Pool-Sum Kernel");
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

def functional_model(x, *, matmul_weight, matmul_bias, max_pool_kernel_size, 
                     max_pool_stride, max_pool_padding, max_pool_dilation, 
                     max_pool_ceil_mode, max_pool_return_indices, scale_factor):
    """
    Optimized functional model bypassing torch.mm and max_pool1d 
    to avoid massive global memory allocations.
    """
    B, I = x.shape
    O = matmul_weight.shape[0]
    device = x.device
    out = torch.zeros(B, device=device)
    
    # Ensure inputs are contiguous float32 buffers
    x = x.contiguous()
    matmul_weight = matmul_weight.contiguous()
    matmul_bias = matmul_bias.contiguous() if matmul_bias is not None else None
    
    fused_ext.fused_op(x, matmul_weight, matmul_bias, out, B, I, O, 
                       max_pool_kernel_size, max_pool_stride, scale_factor)
    return out
