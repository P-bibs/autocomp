# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_074047/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'subtract_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['linear_weight', 'linear_bias', 'subtract_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['linear_weight', 'linear_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a matrix multiplication, subtraction, multiplication, and ReLU activation.
    """

    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

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
    # State for linear (nn.Linear)
    if 'linear_weight' in flat_state:
        state_kwargs['linear_weight'] = flat_state['linear_weight']
    else:
        state_kwargs['linear_weight'] = getattr(model.linear, 'weight', None)
    if 'linear_bias' in flat_state:
        state_kwargs['linear_bias'] = flat_state['linear_bias']
    else:
        state_kwargs['linear_bias'] = getattr(model.linear, 'bias', None)
    if 'subtract_value' in flat_state:
        state_kwargs['subtract_value'] = flat_state['subtract_value']
    else:
        state_kwargs['subtract_value'] = getattr(model, 'subtract_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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
# CUDA kernel source (fused GEMM + element-wise ops)
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile sizes – blockDim = 16×16, chunk size for the K-dimension
constexpr int BLOCK_ROWS = 16;
constexpr int BLOCK_COLS = 16;
constexpr int CHUNK_K   = 64;   // 64 ≤ K (8192) → 128 chunks

__global__ void fused_op_kernel(
    const float* __restrict__ A,   // input  (M × K)
    const float* __restrict__ B,   // weight (N × K)   row-major
    const float* __restrict__ bias,
    float*       __restrict__ C,   // output (M × N)
    const int M, const int N, const int K,
    const float subtract_value,
    const float multiply_value)
{
    // Dynamic shared memory: [A-tile][B-tile]
    extern __shared__ float smem[];
    float* sA = smem;
    float* sB = smem + BLOCK_ROWS * CHUNK_K;

    const int row_start = blockIdx.y * BLOCK_ROWS;
    const int col_start = blockIdx.x * BLOCK_COLS;

    const int row = row_start + threadIdx.y;
    const int col = col_start + threadIdx.x;

    float acc = 0.0f;

    if (row < M && col < N) {
        // Loop over the K-dimension in chunks
        for (int k = 0; k < K; k += CHUNK_K) {
            int tile_k = (K - k) < CHUNK_K ? (K - k) : CHUNK_K;

            // ---- cooperative load of A-tile (BLOCK_ROWS × tile_k) ----
            const int total_threads = BLOCK_ROWS * BLOCK_COLS;
            const int tid = threadIdx.y * BLOCK_COLS + threadIdx.x;

            for (int load_idx = tid;
                 load_idx < BLOCK_ROWS * tile_k;
                 load_idx += total_threads) {
                int i = load_idx / tile_k;          // row within the block
                int j = load_idx % tile_k;          // column (k-offset) within the tile
                int r = row_start + i;              // global row of A
                int kk = k + j;                     // global column of A
                if (r < M && kk < K) {
                    sA[i * CHUNK_K + j] = A[r * K + kk];
                }
            }

            // ---- cooperative load of B-tile (tile_k × BLOCK_COLS) ----
            for (int load_idx = tid;
                 load_idx < tile_k * BLOCK_COLS;
                 load_idx += total_threads) {
                int i = load_idx / BLOCK_COLS;      // k-offset within the tile
                int j = load_idx % BLOCK_COLS;      // column within the block
                int c = col_start + j;              // global column of B (output feature)
                int kk = k + i;                     // global row of B (input feature)
                if (c < N && kk < K) {
                    sB[i * BLOCK_COLS + j] = B[c * K + kk];
                }
            }

            __syncthreads();

            // ---- partial dot product for this chunk ----
            for (int l = 0; l < tile_k; ++l) {
                acc += sA[threadIdx.y * CHUNK_K + l] *
                       sB[l * BLOCK_COLS + threadIdx.x];
            }

            __syncthreads();
        } // next chunk

        // ---- finalise: bias → subtract → multiply → ReLU ----
        float val = acc + bias[col];
        val = (val - subtract_value) * multiply_value;
        if (val < 0.0f) val = 0.0f;
        C[row * N + col] = val;
    }
}

void fused_op(
    torch::Tensor input,   // (M, K)
    torch::Tensor weight,  // (N, K)
    torch::Tensor bias,    // (N)
    torch::Tensor output,  // (M, N) – pre-allocated
    float subtract_value,
    float multiply_value)
{
    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weight.size(0);

    const float* A = input.data_ptr<float>();
    const float* B = weight.data_ptr<float>();
    const float* b = bias.data_ptr<float>();
    float* C = output.data_ptr<float>();

    dim3 block(BLOCK_COLS, BLOCK_ROWS);                 // 16×16 threads
    dim3 grid((N + BLOCK_COLS - 1) / BLOCK_COLS,
              (M + BLOCK_ROWS - 1) / BLOCK_ROWS);

    const int shared_mem = (BLOCK_ROWS * CHUNK_K + CHUNK_K * BLOCK_COLS) * sizeof(float);

    fused_op_kernel<<<grid, block, shared_mem>>>(
        A, B, b, C, M, N, K, subtract_value, multiply_value);

    // CUDA kernel errors are propagated automatically by PyTorch
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float subtract_value,
    float multiply_value);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op,
          "Fused linear + subtract/multiply/ReLU kernel (CUDA)");
}
"""

# -------------------------------------------------------------------------
# Compile the extension (happens at import time)
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# The only entry point that will be imported
# -------------------------------------------------------------------------
def functional_model(
    x,
    *,
    linear_weight,
    linear_bias,
    subtract_value,
    multiply_value,
):
    """
    Fused CUDA kernel:
        y = x @ linear_weight.t() + linear_bias
        y = (y - subtract_value) * multiply_value
        y = relu(y)
    All in a single kernel for maximal throughput.
    """
    # Make sure the inputs reside on the GPU
    if not x.is_cuda:
        x = x.cuda()
    if not linear_weight.is_cuda:
        linear_weight = linear_weight.cuda()
    if not linear_bias.is_cuda:
        linear_bias = linear_bias.cuda()

    M = x.size(0)           # batch size
    K = x.size(1)           # input features
    N = linear_weight.size(0)  # output features

    # Allocate output tensor on the device
    out = torch.empty((M, N), dtype=torch.float32, device='cuda')

    # Launch the fused kernel
    fused_ext.fused_op(
        x,
        linear_weight,
        linear_bias,
        out,
        subtract_value,
        multiply_value,
    )

    return out

batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
