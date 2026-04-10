# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_155300/code_10.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Optimized kernel using shared memory for the input vector 'x'.
// We parallelize the dot product across threads to ensure high occupancy.
__global__ void fused_linear_pool_kernel(
    const float* __restrict__ x, const float* __restrict__ weight, 
    const float* __restrict__ bias, float* __restrict__ out,
    int B, int I, int O, int K, int S, float scale) {
    
    // Each block computes one pool window result for one batch b.
    // The grid is (B * num_pools, 1). 
    // We parallelize the dot product calculation for each neuron in the window.
    
    int tile_id = blockIdx.x;
    int b = tile_id / ((O - K) / S + 1);
    int p = tile_id % ((O - K) / S + 1);
    
    __shared__ float s_partial_dot[256]; 
    
    float window_max = -1e38f;
    
    for (int k_idx = 0; k_idx < K; k_idx++) {
        int o = p * S + k_idx;
        float dot = bias ? bias[o] : 0.0f;
        
        // Parallel dot product
        for (int i = threadIdx.x; i < I; i += blockDim.x) {
            dot += x[b * I + i] * weight[o * I + i];
        }
        
        // Reduce dot product within block
        s_partial_dot[threadIdx.x] = dot;
        __syncthreads();
        
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                s_partial_dot[threadIdx.x] += s_partial_dot[threadIdx.x + offset];
            }
            __syncthreads();
        }
        
        float final_dot = s_partial_dot[0];
        if (final_dot > window_max) window_max = final_dot;
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(&out[b], window_max * scale);
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, 
                      torch::Tensor out, int B, int I, int O, int K, int S, float scale) {
    int num_pools = (O - K) / S + 1;
    dim3 grid(B * num_pools);
    dim3 block(256);
    
    fused_linear_pool_kernel<<<grid, block>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), 
        out.data_ptr<float>(), B, I, O, K, S, scale);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, 
                      torch::Tensor out, int B, int I, int O, int K, int S, float scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Linear Pool Sum");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, matmul_weight, matmul_bias, max_pool_kernel_size, 
                     max_pool_stride, max_pool_padding, max_pool_dilation, 
                     max_pool_ceil_mode, max_pool_return_indices, scale_factor):
    B, I = x.shape
    O = matmul_weight.shape[0]
    
    # Ensure all inputs are contiguous for performance
    x = x.contiguous()
    matmul_weight = matmul_weight.contiguous()
    matmul_bias = matmul_bias.contiguous()
    
    out = torch.zeros(B, device=x.device)
    fused_ext.fused_op(x, matmul_weight, matmul_bias, out, 
                       B, I, O, max_pool_kernel_size, max_pool_stride, scale_factor)
    return out
