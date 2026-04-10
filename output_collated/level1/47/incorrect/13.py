# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_134650/code_27.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs sum reduction over a specified dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

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
    if 'dim' in flat_state:
        state_kwargs['dim'] = flat_state['dim']
    else:
        state_kwargs['dim'] = getattr(model, 'dim')
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

# ------------------------------------------------------------------
# CUDA kernel: Optimized reduction using warp-shuffle
# ------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Warp-level sum reduction for float4
__device__ __forceinline__ void warp_reduce_sum(float4 &val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val.x += __shfl_down_sync(0xffffffff, val.x, offset);
        val.y += __shfl_down_sync(0xffffffff, val.y, offset);
        val.z += __shfl_down_sync(0xffffffff, val.z, offset);
        val.w += __shfl_down_sync(0xffffffff, val.w, offset);
    }
}

extern "C" __global__
void sum_dim1_kernel(const float* __restrict__ input,
                     float*       __restrict__ output,
                     const int B, const int D1, const int D2)
{
    // Each block processes 1 element of the B dimension and 256/4 = 64 elements of D2
    const int b = blockIdx.x;
    const int vec_j_start = blockIdx.y * (blockDim.x / 4); // Threads per block is 256, so 64 vectors per block
    
    // Each thread calculates a partial sum for its assigned slice of D1
    float4 thread_sum = {0.f, 0.f, 0.f, 0.f};
    
    // We assign threads to compute partial sums for a specific vec_j
    int vec_j = vec_j_start + (threadIdx.x / 4);
    int lane = threadIdx.x % 4;
    
    // Ensure we are within bounds
    if (b < B && vec_j * 4 < D2) {
        const float* input_ptr = input + b * (D1 * D2) + vec_j * 4;
        
        // Loop over D1 dimension
        for (int i = 0; i < D1; ++i) {
            float val = input_ptr[i * D2 + lane];
            if (lane == 0) thread_sum.x += val;
            else if (lane == 1) thread_sum.y += val;
            else if (lane == 2) thread_sum.z += val;
            else thread_sum.w += val;
        }

        // Reduce across the 4 threads in the warp that share the same vec_j
        // Each thread in the warp contributes to one of the 4 components. 
        // We use shuffle to communicate across the 4 threads.
        float final_val = 0.0f;
        if (lane == 0) final_val = thread_sum.x;
        else if (lane == 1) final_val = thread_sum.y;
        else if (lane == 2) final_val = thread_sum.z;
        else final_val = thread_sum.w;
        
        // Sum values across lanes 0-3
        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            final_val += __shfl_down_sync(0x0000000F, final_val, offset);
        }
        
        // Broadcast the result to the lane 0 and write to output
        if (lane == 0) {
            output[b * D2 + vec_j * 4] = final_val;
        }
    }
}

void sum_dim1(torch::Tensor input, torch::Tensor output) {
    const int B = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);
    
    // Grid: (B, ceil( (D2/4) / 64 ))
    dim3 threads(256);
    dim3 blocks(B, (D2 / 4 + 63) / 64);
    
    sum_dim1_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void sum_dim1(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1, "Sum along dim 1");
}
"""

sum_ext = load_inline(
    name='sum_dim1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    assert dim == 1
    B, D1, D2 = x.shape
    output = torch.zeros((B, D2), device=x.device, dtype=x.dtype)
    sum_ext.sum_dim1(x, output)
    return output.unsqueeze(1)

def get_init_inputs():
    return [1]

def get_inputs():
    return [torch.rand(128, 4096, 4095, device='cuda')]
