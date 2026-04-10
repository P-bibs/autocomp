# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_103853/code_12.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['A', 'B']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """

    def __init__(self):
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

# -------------------------------------------------------------------------
# CUDA kernel (row‑wise tiled version)
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_forward_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M
) {
    // Tile size = blockDim.x * 4  (256 threads * 4 elements = 1024)
    const int TILE_SIZE = blockDim.x * 4;

    // Number of tiles per row (depends only on M)
    int tilesPerRow = (M + TILE_SIZE - 1) / TILE_SIZE;

    // Which row and which tile does this block handle?
    int blockRow = blockIdx.x / tilesPerRow;
    int tileIdx  = blockIdx.x % tilesPerRow;

    // Guard against possible extra blocks when grid > required
    if (blockRow >= N) return;

    // Start column for this tile
    int colStart = tileIdx * TILE_SIZE;

    // Load A[row] once per block (register)
    float a_val = A[blockRow];

    // Each thread works on up to 4 consecutive elements
    int i = colStart + threadIdx.x * 4;
    if (i >= M) return;   // outside the valid range for this row

    // Fully vectorised path when we have 4 elements left
    if (i + 3 < M) {
        float4 b_vec = reinterpret_cast<const float4*>(B + blockRow * M + i)[0];
        float4 c_vec;
        c_vec.x = a_val * b_vec.x;
        c_vec.y = a_val * b_vec.y;
        c_vec.z = a_val * b_vec.z;
        c_vec.w = a_val * b_vec.w;
        reinterpret_cast<float4*>(C + blockRow * M + i)[0] = c_vec;
    } else {
        // Tail elements (less than 4 remaining)
        for (int j = 0; j < 4; ++j) {
            if (i + j < M) {
                C[blockRow * M + i + j] = a_val * B[blockRow * M + i + j];
            }
        }
    }
}

// Host function that launches the kernel with the correct grid size
void fused_op_forward(const torch::Tensor& A,
                      const torch::Tensor& B,
                      torch::Tensor& C,
                      int N, int M) {
    const int threads = 256;
    // Compute launch configuration:
    //   tile size = 256 threads * 4 elements = 1024
    const int TILE_SIZE = 1024;
    int tiles_per_row = (M + TILE_SIZE - 1) / TILE_SIZE;   // number of tiles per row
    int grid = N * tiles_per_row;                     // total blocks
    // Clamp block count to a safe maximum (CUDA Grid size limit)
    grid = (grid > 65535) ? 65535 : grid;
    fused_op_forward_kernel<<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M);
}
"""

# -------------------------------------------------------------------------
# C++ binding (pybind11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(const torch::Tensor& A,
                      const torch::Tensor& B,
                      torch::Tensor& C,
                      int N, int M);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward,
          "Tile‑based fused unsqueeze‑multiply (row‑wise tiled)");
}
"""

# -------------------------------------------------------------------------
# Build the inline extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional model used for evaluation
# -------------------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs C[i, j] = A[i] * B[i, j] using a row‑wise tiled CUDA kernel.

    Args:
        A: 1‑D tensor of shape (N,)
        B: 2‑D tensor of shape (N, M)

    Returns:
        C: 2‑D tensor of shape (N, M)
    """
    N, M = B.shape
    # Allocate output on the same device
    C = torch.empty(N, M, dtype=torch.float32, device='cuda')

    # Ensure inputs are contiguous and moved to GPU
    A_contig = A.contiguous().cuda()
    B_contig = B.contiguous().cuda()

    # Invoke the custom kernel
    fused_ext.fused_op(A_contig, B_contig, C, N, M)

    return C
