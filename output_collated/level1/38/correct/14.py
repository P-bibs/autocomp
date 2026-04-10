# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_022930/code_14.py
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

# =============================================================================
# CUDA kernel – tiled row processing
# ----------------------------------------------------------------------
# Optimization: Tile operations (item 9). Each block processes TILE_SIZE rows
# sequentially, reducing the number of blocks from N to ⌈N / TILE_SIZE⌉.
# This lowers kernel-launch overhead and improves L2-cache reuse.
# ----------------------------------------------------------------------
cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int TILE_SIZE = 8;               // rows per block

__global__ void fused_normalize_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const int N,
    const int D
) {
    // shared memory for the reduction (one float per thread)
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;

    // base row index for this block
    const int base_row = blockIdx.x * TILE_SIZE;

    // --- process each row in the tile -------------------------------------------------
    for (int r = 0; r < TILE_SIZE; ++r) {
        const int row = base_row + r;
        if (row >= N) return;               // no more rows to handle

        // ----------------------------------------------------------------------
        // Phase 1: compute sum of absolute values for the current row
        // ----------------------------------------------------------------------
        float local_sum = 0.0f;
        for (int d = tid; d < D; d += blockDim.x) {
            local_sum += fabsf(x[row * D + d]);
        }
        sdata[tid] = local_sum;
        __syncthreads();

        // Parallel reduction in shared memory
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        // Normalization factor: D / (sum of |x|)
        const float mean_inv = static_cast<float>(D) / sdata[0];

        // ----------------------------------------------------------------------
        // Phase 2: write the normalized output
        // ----------------------------------------------------------------------
        for (int d = tid; d < D; d += blockDim.x) {
            out[row * D + d] = x[row * D + d] * mean_inv;
        }
        // Ensure all threads finish the write before moving to the next row
        __syncthreads();
    }
}

// Host function that launches the kernel with the proper grid size
void launch_fused_normalize(const at::Tensor& x, at::Tensor& out) {
    const int N = x.size(0);
    const int D = x.size(1);
    const int threads = 512;                 // 512 threads per block (good occupancy)
    const int blocks = (N + TILE_SIZE - 1) / TILE_SIZE;   // one block per tile

    // Dynamic shared memory: one float per thread
    fused_normalize_forward_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D
    );
}
'''

# =============================================================================
# C++ binding (pybind11)
# ----------------------------------------------------------------------
cpp_source = r'''
#include <torch/extension.h>
void launch_fused_normalize(const at::Tensor& x, at::Tensor& out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launch_fused_normalize, "Fused Normalize Forward");
}
'''

# =============================================================================
# Compile the extension
# ----------------------------------------------------------------------
fused_module = load_inline(
    name='fused_normalize',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# =============================================================================
# Functional model used for evaluation
# ----------------------------------------------------------------------
def functional_model(x):
    """
    Tile-optimized fused normalization.
    Receives a (N, D) tensor on the GPU and returns a tensor of the same shape
    where each row is divided by the mean of its absolute values.
    """
    # Ensure row-major layout for coalesced accesses
    if not x.is_contiguous():
        x = x.contiguous()

    out = torch.empty_like(x)          # allocate output on the same device
    fused_module.forward(x, out)       # launch the tiled CUDA kernel
    return out

# -----------------------------------------------------------------------------
# Compatibility glue (not part of the kernel, only for the test harness)
# -----------------------------------------------------------------------------
batch_size = 32768
dim = 65535

def get_init_inputs():
    return []   # no persistent state required

def get_inputs():
    # Random input on the GPU; the test harness will call functional_model
    return [torch.rand(batch_size, dim, device='cuda')]
