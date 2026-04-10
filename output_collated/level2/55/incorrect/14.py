# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160211/code_0.py
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

# Fused CUDA kernel implementing Linear + MaxPool1d + Sum + Scale
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_op_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const int B, const int I, const int O,
    const int K, const int S,
    const float scale) {
    
    const int b = blockIdx.x;  // Batch index
    const int tid = threadIdx.x;  // Thread ID within block
    const int pool_threads = (O - K) / S + 1;  // Number of pooling windows
    
    // Shared memory for partial sums
    extern __shared__ float shared_mem[];
    float* partial_max = shared_mem;  // Shared memory for max values
    
    // Boundary check
    if (b >= B) return;
    
    // Initialize accumulator
    float sum = 0.0f;
    
    // Process each pooling window
    for (int p = tid; p < pool_threads; p += blockDim.x) {
        float window_max = -1e38f;
        
        // Compute max within window
        for (int i = 0; i < K; i++) {
            const int feat_idx = p * S + i;
            if (feat_idx >= O) continue;  // Boundary check
            
            // Compute linear output for this feature
            float dot = (bias) ? bias[feat_idx] : 0.0f;
            const float* x_ptr = x + b * I;
            const float* w_ptr = weight + feat_idx * I;
            
            #pragma unroll 8
            for (int j = 0; j < I; ++j) {
                dot += x_ptr[j] * w_ptr[j];
            }
            
            // Update window maximum
            if (dot > window_max) window_max = dot;
        }
        
        // Store partial max in shared memory
        if (window_max > -1e38f) {
            partial_max[p] = window_max;
        }
    }
    
    __syncthreads();
    
    // Reduce partial maxes (only thread 0 in block performs reduction)
    if (tid == 0) {
        for (int i = 0; i < pool_threads; i++) {
            sum += partial_max[i];
        }
        out[b] = sum * scale;
    }
}

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out,
    int B,
    int I,
    int O,
    int K,
    int S,
    float scale) {
    
    // Configure kernel launch parameters
    const int threads = 256;
    const int blocks = B;
    const int shared_mem_size = ((O - K) / S + 1) * sizeof(float);
    
    // Launch kernel
    fused_op_forward_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        B, I, O, K, S, scale
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out,
    int B,
    int I,
    int O,
    int K,
    int S,
    float scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Linear + MaxPool1d + Sum + Scale");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, matmul_weight, matmul_bias, max_pool_kernel_size, 
                     max_pool_stride, max_pool_padding, max_pool_dilation, 
                     max_pool_ceil_mode, max_pool_return_indices, scale_factor):
    """
    Fused implementation of:
    1. Linear transformation (x @ weight.T + bias)
    2. Max pooling over feature dimension
    3. Sum reduction over pooled dimension
    4. Scaling by scale_factor
    """
    B, I = x.shape
    O = matmul_weight.shape[0]
    
    # Create output tensor
    out = torch.empty(B, dtype=torch.float32, device=x.device)
    
    # Call fused CUDA kernel
    fused_ext.fused_op(
        x, matmul_weight, matmul_bias, out,
        B, I, O,
        max_pool_kernel_size,
        max_pool_stride,
        scale_factor
    )
    
    return out
