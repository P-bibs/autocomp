# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_022930/code_26.py
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

# The strategy is to utilize coalesced reads by having each thread access 
# consecutive memory locations. Instead of one block per row, we process 
# multiple rows per block using 1024 threads for high occupancy.
# We use block-wide reduction for the scalar mean and then normalize.

cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fused_normalize_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const int N,
    const int D
) {
    // Shared memory for reduction: one float per thread
    extern __shared__ float sdata[];
    
    // Each block processes a range of rows.
    // For large D, we process rows in a strided fashion across the grid.
    for (int row = blockIdx.x; row < N; row += gridDim.x) {
        float local_sum = 0.0f;
        const float* row_ptr = x + row * D;
        
        // Coalesced loading of data for summation
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            local_sum += fabsf(row_ptr[d]);
        }
        
        sdata[threadIdx.x] = local_sum;
        __syncthreads();

        // Reduction in shared memory
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
            __syncthreads();
        }

        float total_sum = sdata[0];
        float mean_inv = (float)D / (total_sum + 1e-9f); // Add epsilon for safety

        // Second pass: Final scaling
        float* out_row_ptr = out + row * D;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            out_row_ptr[d] = row_ptr[d] * mean_inv;
        }
        __syncthreads();
    }
}

void launch_fused_normalize(const at::Tensor& x, at::Tensor& out) {
    const int N = x.size(0);
    const int D = x.size(1);
    
    // Config: 1024 threads per block for max occupancy
    const int threads = 1024;
    // Limit blocks to grid occupancy (SM count on 2080Ti is 68, use a multiple)
    const int blocks = std::min(N, 128); 
    
    fused_normalize_forward_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        N, 
        D
    );
}
'''

cpp_source = r'''
#include <torch/extension.h>
void launch_fused_normalize(const at::Tensor& x, at::Tensor& out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launch_fused_normalize, "Fused Normalize Forward");
}
'''

# Compile JIT
fused_module = load_inline(
    name='fused_normalize',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Optimized functional_model:
    Fuses abs(), mean(), and division into a single kernel pass.
    Ensures coalesced memory access with thread-independent reduction.
    """
    if not x.is_contiguous():
        x = x.contiguous()
        
    out = torch.empty_like(x)
    fused_module.forward(x, out)
    return out
