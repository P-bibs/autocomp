# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_022930/code_6.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs L1 normalization.
    """

    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(ModelNew, self).__init__()

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

# Vectorized CUDA kernel for fused normalize operation
cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vectorized_normalize_kernel(const float* __restrict__ x, float* __restrict__ out, int N, int D) {
    extern __shared__ float sdata[];
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    if (bid >= N) return;

    // Vectorized Phase: Accumulate absolute sums
    float local_sum = 0.0f;
    const float4* x4 = (const float4*)x;
    int D_vec = D / 4;  // Number of float4 elements per row
    int remainder = D % 4;  // Handle non-multiple-of-4 remainder

    // Process vectorized elements (float4)
    for (int d = tid; d < D_vec; d += blockDim.x) {
        float4 val = x4[bid * D_vec + d];
        local_sum += (fabsf(val.x) + fabsf(val.y) + fabsf(val.z) + fabsf(val.w));
    }

    // Handle remainder elements that don't fit into float4
    if (remainder > 0) {
        for (int d = D_vec * 4 + tid; d < D; d += blockDim.x) {
            local_sum += fabsf(x[bid * D + d]);
        }
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Compute reciprocal of mean absolute value
    float mean_inv = (float)D / sdata[0];
    
    float4* out4 = (float4*)out;

    // Vectorized Phase: Normalize and store results
    for (int d = tid; d < D_vec; d += blockDim.x) {
        float4 val = x4[bid * D_vec + d];
        val.x *= mean_inv;
        val.y *= mean_inv;
        val.z *= mean_inv;
        val.w *= mean_inv;
        out4[bid * D_vec + d] = val;
    }

    // Handle remainder elements
    if (remainder > 0) {
        for (int d = D_vec * 4 + tid; d < D; d += blockDim.x) {
            out[bid * D + d] = x[bid * D + d] * mean_inv;
        }
    }
}

void launch_vectorized_normalize(const at::Tensor& x, at::Tensor& out) {
    int N = x.size(0);
    int D = x.size(1);
    const int threads = 256;  // Optimal for modern GPUs
    
    // Launch N blocks, one per row
    vectorized_normalize_kernel<<<N, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D
    );
}
'''

# C++ interface to expose the kernel to Python
cpp_source = r'''
#include <torch/extension.h>

void launch_vectorized_normalize(const at::Tensor& x, at::Tensor& out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launch_vectorized_normalize, "Vectorized Normalize Forward");
}
'''

# Compile the extension with optimizations
fused_module = load_inline(
    name='vectorized_normalize',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Optimized vectorized version of the normalize function.
    Fuses abs(), mean(), and division into a single kernel pass using float4 vectorization.
    """
    # Ensure contiguous memory layout for optimal memory access
    if not x.is_contiguous():
        x = x.contiguous()
        
    out = torch.empty_like(x)
    fused_module.forward(x, out)
    return out

# --- Compatibility requirements ---
batch_size = 32768
dim = 65535

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]
