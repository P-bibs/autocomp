# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_153614/code_4.py
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

# Define the optimized CUDA kernel
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

#define TILE_I 32
#define THREADS 256

__global__ void fused_op_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int B, int I, int O,
    int K, int S, float scale)
{
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int k = tid;

    if (k >= O) return;

    // Load bias into register
    float dot = (bias != nullptr) ? bias[k] : 0.0f;

    // Shared memory for tiling
    extern __shared__ float shmem[];
    float* sh_x = shmem;
    float* sh_w = shmem + TILE_I;

    // Tiled GEMM
    for (int i0 = 0; i0 < I; i0 += TILE_I) {
        // Coalesced load of input tile
        if (tid < TILE_I && i0 + tid < I) {
            sh_x[tid] = x[b * I + i0 + tid];
        }
        
        // Coalesced load of weight tile for this thread's output feature
        if (i0 + tid < I) {
            sh_w[tid] = w[k * I + i0 + tid];
        }
        __syncthreads();

        // Compute partial dot product
        #pragma unroll 4
        for (int j = 0; j < TILE_I && i0 + j < I; ++j) {
            dot += sh_x[j] * sh_w[j];
        }
        __syncthreads();
    }

    // Max-pooling and sum
    const int num_windows = (O - K) / S + 1;
    float total_sum = 0.0f;

    for (int w_idx = 0; w_idx < num_windows; ++w_idx) {
        const int win_start = w_idx * S;
        const int win_end = win_start + K;

        bool in_window = (k >= win_start) && (k < win_end);
        float my_val = in_window ? dot : -FLT_MAX;

        // Warp-level reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            my_val = fmaxf(my_val, __shfl_down_sync(0xffffffff, my_val, offset));

        // Inter-warp reduction using shared memory
        if (tid % 32 == 0) {
            sh_x[tid / 32] = my_val;
        }
        __syncthreads();

        if (tid < 8) {
            my_val = sh_x[tid];
            #pragma unroll
            for (int offset = 4; offset > 0; offset >>= 1)
                my_val = fmaxf(my_val, __shfl_down_sync(0xffffffff, my_val, offset));
            if (tid == 0) {
                total_sum += my_val;
            }
        }
        __syncthreads();
    }

    // Write final result
    if (tid == 0) {
        out[b] = total_sum * scale;
    }
}

void fused_op_forward(
    int blocks, int threads,
    torch::Tensor x, torch::Tensor w, torch::Tensor bias,
    torch::Tensor out, int B, int I, int O,
    int K, int S, float scale,
    int shared_mem)
{
    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = w.data_ptr<float>();
    const float* bias_ptr = bias.numel() ? bias.data_ptr<float>() : nullptr;
    float* out_ptr = out.data_ptr<float>();

    fused_op_forward_kernel<<<blocks, threads, shared_mem>>>(
        x_ptr, w_ptr, bias_ptr, out_ptr,
        B, I, O, K, S, scale
    );
}
"""

# C++ binding
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    int blocks, int threads,
    torch::Tensor x, torch::Tensor w, torch::Tensor bias,
    torch::Tensor out, int B, int I, int O,
    int K, int S, float scale,
    int shared_mem);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused linear-maxpool-sum kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def fused_op_forward(x, w, bias, scale, K, S):
    B, I = x.shape
    O, _ = w.shape
    out = torch.empty(B, dtype=x.dtype, device=x.device)
    threads = 256
    blocks = B
    shared_mem = (TILE_I + TILE_I * threads) * 4  # TILE_I floats for x + TILE_I*threads floats for w
    
    fused_ext.fused_op(
        blocks, threads,
        x, w, bias if bias is not None else torch.tensor([], dtype=x.dtype, device=x.device),
        out, B, I, O, K, S, float(scale),
        shared_mem
    )
    return out

def functional_model(x, *, matmul_weight, matmul_bias,
                     max_pool_kernel_size, max_pool_stride,
                     max_pool_padding, max_pool_dilation,
                     max_pool_ceil_mode, max_pool_return_indices,
                     scale_factor):
    # padding/dilation/ceil_mode/return_indices are ignored because the
    # original model used a plain 1-D max-pool without those features.
    return fused_op_forward(
        x.contiguous(),
        matmul_weight.contiguous(),
        matmul_bias.contiguous() if matmul_bias is not None else None,
        scale_factor,
        max_pool_kernel_size,
        max_pool_stride
    )
