# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_151353/code_3.py
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

# -------------------------------------------------------------------------
# CUDA source – fused kernel implementation
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_op_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int B,
    const int I,
    const int O,
    const int kernel_size,
    const int stride,
    const float scale_factor,
    float* __restrict__ output
) {
    const int b = blockIdx.x;  // batch index
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // Shared memory for reduction
    extern __shared__ float sdata[];
    
    // Compute linear transformation and max pool in one pass
    const int num_windows = (O - kernel_size) / stride + 1;
    const int windows_per_block = (num_windows + gridDim.y - 1) / gridDim.y;
    const int window_start_idx = (blockIdx.y * windows_per_block) * stride;
    
    float local_sum = 0.0f;
    
    // Each thread processes multiple pooling windows
    for (int w_base = 0; w_base < windows_per_block; w_base++) {
        const int win_idx = blockIdx.y * windows_per_block + w_base;
        if (win_idx >= num_windows) break;
        
        const int start = win_idx * stride;
        float max_val = -INFINITY;
        
        // Compute max pooling directly from linear results
        for (int i = 0; i < kernel_size; i++) {
            const int k = start + i;
            if (k < O) {
                float val = bias[k];
                // Compute dot product for this output dimension
                for (int j = tid; j < I; j += block_size) {
                    val += x[b * I + j] * weight[k * I + j];
                }
                
                // Reduction to get the final value for this dimension
                for (int stride_sz = block_size / 2; stride_sz > 0; stride_sz >>= 1) {
                    float temp = 0.0f;
                    if (tid < stride_sz && (tid + stride_sz) < I) {
                        // This is a simplified reduction - in practice, you'd need a proper tree reduction
                        // or use shared memory. For now, we'll do a simpler approach.
                    }
                }
                
                // Since each thread only computes partial sums, we need to reduce across threads
                // We'll use a two-phase approach: first compute partial results, then reduce
                
                // Use shared memory to reduce the dot product result
                __shared__ float shared_val[256]; // Assumes max 256 threads per block
                shared_val[tid] = val;
                __syncthreads();
                
                // Reduction in shared memory
                for (int s = block_size / 2; s > 0; s >>= 1) {
                    if (tid < s) {
                        shared_val[tid] += shared_val[tid + s];
                    }
                    __syncthreads();
                }
                
                val = shared_val[0];
                __syncthreads();
                
                if (val > max_val) max_val = val;
            }
        }
        local_sum += max_val;
    }
    
    // Store local sum in shared memory for block-level reduction
    sdata[tid] = local_sum;
    __syncthreads();
    
    // Block-level reduction
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result of this block to global memory
    if (tid == 0) {
        atomicAdd(&output[b], sdata[0] * scale_factor);
    }
}

// More efficient implementation that separates linear computation from pooling
__global__ void linear_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int B,
    const int I,
    const int O,
    float* __restrict__ linear_output
) {
    const int b = blockIdx.x;
    const int k = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (k >= O) return;
    
    float sum = (bias != nullptr) ? bias[k] : 0.0f;
    const float* x_row = x + b * I;
    const float* w_row = weight + k * I;
    
    // Vectorized reduction for better memory coalescing
    for (int i = threadIdx.x; i < I; i += blockDim.x) {
        sum += x_row[i] * w_row[i];
    }
    
    // Reduction within warp first
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    // Then reduce using shared memory across the block
    __shared__ float sdata[256];
    if (threadIdx.x < 32) {
        sdata[threadIdx.x] = sum;
    }
    __syncthreads();
    
    if (threadIdx.x < 16) {
        sdata[threadIdx.x] += sdata[threadIdx.x + 16];
        sdata[threadIdx.x] += sdata[threadIdx.x + 8];
        sdata[threadIdx.x] += sdata[threadIdx.x + 4];
        sdata[threadIdx.x] += sdata[threadIdx.x + 2];
        sdata[threadIdx.x] += sdata[threadIdx.x + 1];
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        linear_output[b * O + k] = sdata[0];
    }
}

__global__ void pool_and_sum_kernel(
    const float* __restrict__ linear_output,
    const int B,
    const int O,
    const int kernel_size,
    const int stride,
    const float scale_factor,
    float* __restrict__ output
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // Number of pooled windows
    const int L = (O - kernel_size) / stride + 1;
    
    // Windows handled by this block
    const int windows_per_block = (L + gridDim.y - 1) / gridDim.y;
    const int win_start = blockIdx.y * windows_per_block;
    const int win_end = min(win_start + windows_per_block, L);
    
    float partial_sum = 0.0f;
    
    for (int win_idx = win_start; win_idx < win_end; win_idx++) {
        const int start = win_idx * stride;
        float max_val = -INFINITY;
        
        for (int i = 0; i < kernel_size; i++) {
            const int idx = b * O + start + i;
            float val = linear_output[idx];
            if (val > max_val) max_val = val;
        }
        partial_sum += max_val;
    }
    
    // Shared memory for block reduction
    extern __shared__ float sdata[];
    sdata[tid] = partial_sum;
    __syncthreads();
    
    // Reduction within block
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(&output[b], sdata[0] * scale_factor);
    }
}

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int B, int I, int O,
    int max_pool_kernel_size,
    int max_pool_stride,
    float scale_factor,
    torch::Tensor output
) {
    // Allocate temporary buffer for linear output
    auto linear_output = torch::empty({B, O}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int BLOCK_SIZE = 256;
    
    // Launch linear kernel
    dim3 linear_grid(B, (O + BLOCK_SIZE - 1) / BLOCK_SIZE);
    linear_kernel<<<linear_grid, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        B, I, O,
        linear_output.data_ptr<float>()
    );
    
    // Initialize output to zero since we're using atomic adds
    output.zero_();
    
    // Calculate number of windows
    const int L = (O - max_pool_kernel_size) / max_pool_stride + 1;
    
    // Launch pool and sum kernel
    dim3 pool_grid(B, min(L, 65535)); // Limit grid size
    const int shared_mem_size = BLOCK_SIZE * sizeof(float);
    pool_and_sum_kernel<<<pool_grid, BLOCK_SIZE, shared_mem_size>>>(
        linear_output.data_ptr<float>(),
        B, O,
        max_pool_kernel_size,
        max_pool_stride,
        scale_factor,
        output.data_ptr<float>()
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int B, int I, int O,
    int max_pool_kernel_size,
    int max_pool_stride,
    float scale_factor,
    torch::Tensor output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused linear + max-pool + sum + scale operation");
}
"""

# -------------------------------------------------------------------------
# Build the inline extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name="fused_op_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    with_cuda=True,
)

# -------------------------------------------------------------------------
# Problem parameters (same as the original script)
# -------------------------------------------------------------------------
batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]

def get_inputs():
    return [torch.rand(batch_size, in_features)]

# -------------------------------------------------------------------------
# The fused functional_model that will be imported during evaluation
# -------------------------------------------------------------------------
def functional_model(
    x,
    *,
    matmul_weight,
    matmul_bias,
    max_pool_kernel_size,
    max_pool_stride,
    max_pool_padding,       # unused (assumed 0)
    max_pool_dilation,      # unused (assumed 1)
    max_pool_ceil_mode,     # unused
    max_pool_return_indices,# unused
    scale_factor,
):
    # Ensure all inputs are on GPU
    if not x.is_cuda:
        x = x.cuda()
    if not matmul_weight.is_cuda:
        matmul_weight = matmul_weight.cuda()
    if matmul_bias is not None and not matmul_bias.is_cuda:
        matmul_bias = matmul_bias.cuda()
    
    # Get dimensions
    B = x.size(0)  # batch size
    I = x.size(1)  # input features
    O = matmul_weight.size(0)  # output features
    
    # Create output tensor
    output = torch.empty(B, device="cuda", dtype=torch.float32)
    
    # Call fused operation
    fused_ext.fused_op(
        x,
        matmul_weight,
        matmul_bias if matmul_bias is not None else torch.empty(0, device="cuda"),
        B, I, O,
        max_pool_kernel_size,
        max_pool_stride,
        scale_factor,
        output
    )
    
    return output
