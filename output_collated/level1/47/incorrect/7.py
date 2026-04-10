# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_131936/code_1.py
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
#  CUDA source – vectorized reduction kernel with warp-level optimizations
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Helper function for warp-level reduction
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Vectorized kernel using float4 + warp-level reductions
__global__ void sum_dim1_float4_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       const int B, const int D1, const int D2) {
    const int tid = threadIdx.x;
    const int b = blockIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int elements_per_thread = 4;
    const int threads_per_block = blockDim.x;
    const int warps_per_block = (threads_per_block + 31) / 32;
    
    if (b >= B) return;

    // Shared memory for partial sums from each warp
    extern __shared__ float sdata[];
    float* warp_sums = sdata;

    // Calculate the base index for this thread
    const int elements_per_block = blockDim.x * elements_per_thread;
    const int j_base = blockIdx.y * elements_per_block + tid * elements_per_thread;

    // Initialize sums
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    // Process data with float4 when aligned
    if ((D2 % 4 == 0) && (j_base + 3 < D2)) {
        // Aligned case - use float4 for better memory coalescing
        for (int i = 0; i < D1; ++i) {
            const float4 val = __ldg((const float4*)&input[b * D1 * D2 + i * D2 + j_base]);
            sum0 += val.x;
            sum1 += val.y;
            sum2 += val.z;
            sum3 += val.w;
        }
    } else {
        // Unaligned case - use scalar loads with bounds checking
        for (int i = 0; i < D1; ++i) {
            const int base_offset = b * D1 * D2 + i * D2 + j_base;
            if (j_base + 0 < D2) sum0 += __ldg(&input[base_offset + 0]);
            if (j_base + 1 < D2) sum1 += __ldg(&input[base_offset + 1]);
            if (j_base + 2 < D2) sum2 += __ldg(&input[base_offset + 2]);
            if (j_base + 3 < D2) sum3 += __ldg(&input[base_offset + 3]);
        }
    }

    // Perform warp-level reductions on each component
    sum0 = warp_reduce_sum(sum0);
    sum1 = warp_reduce_sum(sum1);
    sum2 = warp_reduce_sum(sum2);
    sum3 = warp_reduce_sum(sum3);

    // Store warp results in shared memory
    if (lane_id == 0) {
        const int warp_offset = warp_id * 4;
        if (warp_offset + 0 < warps_per_block * 4) warp_sums[warp_offset + 0] = sum0;
        if (warp_offset + 1 < warps_per_block * 4) warp_sums[warp_offset + 1] = sum1;
        if (warp_offset + 2 < warps_per_block * 4) warp_sums[warp_offset + 2] = sum2;
        if (warp_offset + 3 < warps_per_block * 4) warp_sums[warp_offset + 3] = sum3;
    }

    __syncthreads();

    // Thread 0 in each warp does the final reduction and writes to global memory
    if (tid < 32 && lane_id == 0) {
        float final_sum0 = 0.0f, final_sum1 = 0.0f, final_sum2 = 0.0f, final_sum3 = 0.0f;
        
        // Accumulate results from all warps
        for (int w = 0; w < warps_per_block; ++w) {
            const int warp_offset = w * 4;
            if (warp_offset + 0 < warps_per_block * 4) final_sum0 += warp_sums[warp_offset + 0];
            if (warp_offset + 1 < warps_per_block * 4) final_sum1 += warp_sums[warp_offset + 1];
            if (warp_offset + 2 < warps_per_block * 4) final_sum2 += warp_sums[warp_offset + 2];
            if (warp_offset + 3 < warps_per_block * 4) final_sum3 += warp_sums[warp_offset + 3];
        }

        // Write final results to global memory
        const int output_base = b * D2 + blockIdx.y * elements_per_block + tid * elements_per_thread;
        if (output_base + 0 < b * D2 + D2) output[output_base + 0] = final_sum0;
        if (output_base + 1 < b * D2 + D2) output[output_base + 1] = final_sum1;
        if (output_base + 2 < b * D2 + D2) output[output_base + 2] = final_sum2;
        if (output_base + 3 < b * D2 + D2) output[output_base + 3] = final_sum3;
    }
}

// Host-side wrapper
void sum_dim1_float4(torch::Tensor input, torch::Tensor output) {
    const int B = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);

    const int threads = 256;  // Using 256 threads for better occupancy on RTX 2080Ti
    const int elements_per_thread = 4;
    const int elements_per_block = threads * elements_per_thread;
    
    const int blocks_x = B;
    const int blocks_y = (D2 + elements_per_block - 1) / elements_per_block;
    
    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads);

    // Shared memory size: each warp can produce 4 partial sums
    const int warps_per_block = (threads + 31) / 32;
    const int shared_mem_size = warps_per_block * 4 * sizeof(float);

    sum_dim1_float4_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2);
}
"""

# -------------------------------------------------------------------------
#  C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void sum_dim1_float4(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1_float4", &sum_dim1_float4,
          "Optimized sum along dimension 1 using float4 and warp reductions");
}
"""

# Compile the extension
sum_ext_float4 = load_inline(
    name='sum_dim1_float4',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
#  Functional model – uses the custom float4-optimized kernel
# -------------------------------------------------------------------------
def functional_model(x, *, dim):
    """
    Reduce input tensor `x` of shape (B, D1, D2) along dimension 1 using a
    custom CUDA kernel optimized with float4 memory loads and warp-level reductions.
    """
    assert dim == 1, "Only reduction along dimension 1 is supported."
    
    B, D1, D2 = x.shape
    
    # Output tensor with shape (B, D2)
    output = torch.zeros((B, D2), device=x.device, dtype=x.dtype)
    
    # Launch the optimized kernel
    sum_ext_float4.sum_dim1_float4(x, output)
    
    # Reshape to (B, 1, D2) to match expected output format
    return output.unsqueeze(1)

# -------------------------------------------------------------------------
#  Quick sanity-check when the file is executed directly
# -------------------------------------------------------------------------
if __name__ == "__main__":
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    reduce_dim = 1

    x = torch.rand(batch_size, dim1, dim2, device='cuda', dtype=torch.float32)
    out = functional_model(x, dim=reduce_dim)

    # Shape check
    assert out.shape == (batch_size, 1, dim2), f"Wrong shape: {out.shape}"

    # Result check against PyTorch's native reduction
    ref = x.sum(dim=1, keepdim=True)
    assert torch.allclose(out, ref, atol=1e-4), "Results diverge!"
    print("✅ functional_model with optimized warp-level kernel works correctly.")
