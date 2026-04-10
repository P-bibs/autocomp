# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152808/code_2.py
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
#include <cuda_fp16.h>

#define TILE_SIZE 256
#define WARP_SIZE 32

// Optimized kernel using shared memory tiling and coalesced access
__global__ void fused_linear_pool_sum_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int B, int I, int O, int K, int S, float scale) {
    
    int b = blockIdx.x;
    int p = blockIdx.y; // Pool window index
    int tid = threadIdx.x;
    
    // Shared memory for input vector (broadcast to all threads)
    extern __shared__ float s_x[];
    
    // Load input vector into shared memory
    for (int i = tid; i < I; i += blockDim.x) {
        s_x[i] = x[b * I + i];
    }
    __syncthreads();
    
    float window_max = -1e38f;
    
    // Process each element in the pooling window
    for (int k_idx = 0; k_idx < K; ++k_idx) {
        int feat_idx = p * S + k_idx;
        if (feat_idx >= O) continue;
        
        float dot = (bias != nullptr) ? bias[feat_idx] : 0.0f;
        
        // Compute dot product using coalesced access pattern
        for (int i = 0; i < I; ++i) {
            dot += s_x[i] * weight[feat_idx * I + i];
        }
        
        if (dot > window_max) {
            window_max = dot;
        }
    }
    
    // Use shared memory for reduction within block
    __shared__ float s_max_vals[TILE_SIZE];
    s_max_vals[tid] = window_max;
    __syncthreads();
    
    // Reduce max values within warp
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        if (tid < offset) {
            s_max_vals[tid] = fmaxf(s_max_vals[tid], s_max_vals[tid + offset]);
        }
        __syncthreads();
    }
    
    // First thread of each warp writes to shared memory
    if (tid % WARP_SIZE == 0) {
        s_max_vals[tid/WARP_SIZE] = s_max_vals[tid];
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (tid < WARP_SIZE) {
        float final_max = -1e38f;
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        for (int i = 0; i < num_warps; ++i) {
            final_max = fmaxf(final_max, s_max_vals[i]);
        }
        
        // Atomic add to global accumulator
        if (final_max != -1e38f) {
            atomicAdd(&out[b], final_max);
        }
    }
}

void fused_op_forward(int B, int I, int O, int K, int S, float scale,
                      const float* x, const float* weight, const float* bias, float* out) {
    // Initialize output to zero
    cudaMemset(out, 0, B * sizeof(float));
    
    dim3 grid(B, (O - K) / S + 1);
    dim3 block(TILE_SIZE);
    
    // Shared memory size for input vector
    int shared_mem_size = I * sizeof(float);
    
    fused_linear_pool_sum_kernel<<<grid, block, shared_mem_size>>>(
        x, weight, bias, out, B, I, O, K, S, scale);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(int B, int I, int O, int K, int S, float scale,
                      const float* x, const float* weight, const float* bias, float* out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Linear-Pool-Sum operation");
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
    B = x.shape[0]
    I = x.shape[1]
    O = matmul_weight.shape[0]
    K = max_pool_kernel_size
    S = max_pool_stride
    
    out = torch.zeros(B, device=x.device, dtype=torch.float32)
    
    fused_ext.fused_op(B, I, O, K, S, scale_factor,
                       x.data_ptr<float>(), 
                       matmul_weight.data_ptr<float>(), 
                       matmul_bias.data_ptr<float>() if matmul_bias is not None else 0,
                       out.data_ptr<float>())
    
    return out
