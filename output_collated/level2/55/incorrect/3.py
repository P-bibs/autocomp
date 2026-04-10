# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_151918/code_14.py
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
# CUDA implementation
# Fuses Linear, MaxPool, and Sum into high-throughput kernels.
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float atomicMaxFloat(float* addr, float val) {
    unsigned int* addr_as_uint = (unsigned int*)addr;
    unsigned int old = *addr_as_uint;
    unsigned int assumed;
    do {
        assumed = old;
        float old_val = __uint_as_float(assumed);
        if (val <= old_val) return old_val;
        old = atomicCAS(addr_as_uint, assumed, __float_as_uint(val));
    } while (assumed != old);
    return __uint_as_float(assumed);
}

__global__ void fused_linear_pool_kernel(
    const float* __restrict__ x,
    const float* __restrict__ W,
    const float* __restrict__ b,
    float* __restrict__ wmax,
    const int B, const int I, const int O,
    const int K, const int S) {

    const int b_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_windows = (O - K) / S + 1;

    // Grid-stride loop over features O
    for (int k = tid; k < O; k += blockDim.x) {
        float val = (b != nullptr) ? b[k] : 0.0f;
        const float* x_ptr = x + b_idx * I;
        const float* w_ptr = W + k * I;

        #pragma unroll 8
        for (int i = 0; i < I; ++i) {
            val += __ldg(&x_ptr[i]) * __ldg(&w_ptr[i]);
        }

        int start = (k >= K) ? ((k - K + 1) / S) : 0;
        int end = (k / S) + 1;
        if (end > num_windows) end = num_windows;

        for (int w = start; w < end; ++w) {
            atomicMaxFloat(&wmax[b_idx * num_windows + w], val);
        }
    }
}

__global__ void reduce_sum_kernel(
    const float* __restrict__ wmax,
    float* __restrict__ out,
    const int B,
    const int num_windows,
    const float scale) {

    const int b_idx = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float sdata[];

    float sum = 0.0f;
    for (int i = tid; i < num_windows; i += blockDim.x) {
        sum += wmax[b_idx * num_windows + i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[b_idx] = sdata[0] * scale;
}

void fused_op_forward(torch::Tensor x, torch::Tensor W, torch::Tensor b, 
                      torch::Tensor out, int B, int I, int O, int K, int S, float scale) {
    int num_windows = (O - K) / S + 1;
    auto wmax = torch::full({B, num_windows}, -1e38f, x.options());

    // Launch Fused Kernel
    fused_linear_pool_kernel<<<B, 256>>>(
        x.data_ptr<float>(), W.data_ptr<float>(), 
        b.defined() ? b.data_ptr<float>() : nullptr,
        wmax.data_ptr<float>(), B, I, O, K, S);

    // Launch Reduction Kernel
    reduce_sum_kernel<<<B, 256, 256 * sizeof(float)>>>(
        wmax.data_ptr<float>(), out.data_ptr<float>(), B, num_windows, scale);
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor x, torch::Tensor W, torch::Tensor b, 
                      torch::Tensor out, int B, int I, int O, int K, int S, float scale);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward);
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
    K, S = max_pool_kernel_size, max_pool_stride
    out = torch.empty(B, device=x.device, dtype=x.dtype)
    fused_ext.fused_op_forward(x, matmul_weight, matmul_bias, out, B, I, O, K, S, scale_factor)
    return out
