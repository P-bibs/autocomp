# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_125421/code_10.py
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

# -------------------------------------------------------------------------
#  High-performance CUDA implementation with warp-level primitives
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_dim1_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int B, int D1, int D2) {
    // One warp (32 threads) is responsible for one output element (b, j)
    const int warp_id   = threadIdx.x / 32;                // warp index inside the block
    const int lane_id   = threadIdx.x % 32;                // lane inside the warp
    const int j_global  = blockIdx.x * (blockDim.x/32) + warp_id;   // column index
    const int b         = blockIdx.y;                     // batch index

    if (b >= B || j_global >= D2) return;

    // Pointer to the first element of this (b, :, j) column
    const float* col_ptr = input + (b * D1 * D2) + j_global;

    // -----------------------------------------------------------------
    // 1) each lane reads a strided subset of the D1 dimension
    // -----------------------------------------------------------------
    float thread_sum = 0.0f;
    // stride = warpSize (32) because successive lanes read successive i-values
    for (int i = lane_id; i < D1; i += 32) {
        thread_sum += col_ptr[i * D2];        // coalesced: adjacent lanes read adjacent i
    }

    // -----------------------------------------------------------------
    // 2) intra-warp reduction using shuffle (no shared memory)
    // -----------------------------------------------------------------
    // mask for the whole warp
    const unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // -----------------------------------------------------------------
    // 3) lane 0 writes the final sum
    // -----------------------------------------------------------------
    if (lane_id == 0) {
        output[b * D2 + j_global] = thread_sum;
    }
}

void sum_dim1_gpu(torch::Tensor input, torch::Tensor output) {
    int B = input.size(0);
    int D1 = input.size(1);
    int D2 = input.size(2);
    
    // Threads per block: choose 256 for a balance of occupancy
    // Interpreted as 8 warps per block
    const int warps_per_block = 8;
    dim3 block(warps_per_block * 32);
    dim3 grid((D2 + warps_per_block - 1) / warps_per_block, B);
    
    sum_dim1_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void sum_dim1_gpu(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1_gpu", &sum_dim1_gpu, "Optimized sum along dim 1 with warp primitives");
}
"""

# Compile the optimized extension
sum_ext = load_inline(
    name='sum_dim1_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Sum along dimension 1 using custom optimized CUDA kernel with warp-level primitives.
    Shape: (B, D1, D2) -> (B, 1, D2)
    """
    assert dim == 1, "Only dim=1 is supported."
    
    batch, d1, d2 = x.shape
    output = torch.zeros((batch, d2), device=x.device, dtype=x.dtype)
    
    # Kernel computes (B, D2) result
    sum_ext.sum_dim1_gpu(x, output)
    
    return output.view(batch, 1, d2)

if __name__ == "__main__":
    # Sanity check
    batch_size, d1, d2 = 32, 128, 64
    x = torch.randn(batch_size, d1, d2, device='cuda')
    expected = x.sum(dim=1, keepdim=True)
    actual = functional_model(x, dim=1)
    
    assert torch.allclose(actual, expected, atol=1e-5)
    print("Optimization successful: output matches torch.sum.")
