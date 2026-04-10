# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_145711/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'eps', 'momentum']
FORWARD_ARG_NAMES = ['x', 'y']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['bmm_weight', 'bmm_bias', 'instance_norm_running_mean', 'instance_norm_running_var', 'instance_norm_weight', 'instance_norm_bias', 'instance_norm_use_input_stats', 'instance_norm_momentum', 'instance_norm_eps']
REQUIRED_FLAT_STATE_NAMES = ['bmm_weight', 'bmm_bias', 'instance_norm_running_mean', 'instance_norm_running_var', 'instance_norm_weight', 'instance_norm_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a batch matrix multiplication, instance normalization, summation, residual addition, and multiplication.
    """

    def __init__(self, in_features, out_features, eps=1e-05, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)

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
    # State for bmm (nn.Linear)
    if 'bmm_weight' in flat_state:
        state_kwargs['bmm_weight'] = flat_state['bmm_weight']
    else:
        state_kwargs['bmm_weight'] = getattr(model.bmm, 'weight', None)
    if 'bmm_bias' in flat_state:
        state_kwargs['bmm_bias'] = flat_state['bmm_bias']
    else:
        state_kwargs['bmm_bias'] = getattr(model.bmm, 'bias', None)
    # State for instance_norm (nn.InstanceNorm2d)
    if 'instance_norm_running_mean' in flat_state:
        state_kwargs['instance_norm_running_mean'] = flat_state['instance_norm_running_mean']
    else:
        state_kwargs['instance_norm_running_mean'] = getattr(model.instance_norm, 'running_mean', None)
    if 'instance_norm_running_var' in flat_state:
        state_kwargs['instance_norm_running_var'] = flat_state['instance_norm_running_var']
    else:
        state_kwargs['instance_norm_running_var'] = getattr(model.instance_norm, 'running_var', None)
    if 'instance_norm_weight' in flat_state:
        state_kwargs['instance_norm_weight'] = flat_state['instance_norm_weight']
    else:
        state_kwargs['instance_norm_weight'] = getattr(model.instance_norm, 'weight', None)
    if 'instance_norm_bias' in flat_state:
        state_kwargs['instance_norm_bias'] = flat_state['instance_norm_bias']
    else:
        state_kwargs['instance_norm_bias'] = getattr(model.instance_norm, 'bias', None)
    state_kwargs['instance_norm_use_input_stats'] = not model.instance_norm.track_running_stats
    state_kwargs['instance_norm_momentum'] = model.instance_norm.momentum
    state_kwargs['instance_norm_eps'] = model.instance_norm.eps
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --------------------------------------------------------------
# 1. CUDA kernel (GEMM + bias) – tiled shared-memory version
# --------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// Tile-based GEMM kernel: C = A (M×K) * B (N×K)ᵀ + bias
__global__ void gemm_forward_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    const float* __restrict__ bias,
                                    float* __restrict__ C,
                                    int M, int N, int K)
{
    // Thread indices
    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x; // ∈ [0, M)
    int col = blockIdx.y * BLOCK_SIZE + threadIdx.y; // ∈ [0, N)

    // Shared memory for tiles of A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    // Number of tiles along the K dimension
    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tile = 0; tile < numTiles; ++tile) {
        // ---- load tile from A (row fixed, column = tile*BLOCK_SIZE + threadIdx.y) ----
        int aCol = tile * BLOCK_SIZE + threadIdx.y;
        if (row < M && aCol < K) {
            As[threadIdx.x][threadIdx.y] = A[row * K + aCol];
        } else {
            As[threadIdx.x][threadIdx.y] = 0.0f;
        }

        // ---- load tile from B (column fixed, row = tile*BLOCK_SIZE + threadIdx.x) ----
        // B is stored as (N, K) → weight[out_feature][in_feature]
        int bRow = tile * BLOCK_SIZE + threadIdx.x;
        if (bRow < K && col < N) {
            // B[col * K + bRow] == weight[out_feature = col][in_feature = bRow]
            Bs[threadIdx.x][threadIdx.y] = B[col * K + bRow];
        } else {
            Bs[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        // ---- compute partial dot product for this tile ----
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += As[threadIdx.x][i] * Bs[i][threadIdx.y];
        }

        __syncthreads();
    }

    // ---- write result and add bias ----
    if (row < M && col < N) {
        float val = sum;
        if (bias != nullptr) {
            val += bias[col];
        }
        C[row * N + col] = val;
    }
}

// Host wrapper that launches the kernel
void gemm_forward(const torch::Tensor& A,
                  const torch::Tensor& B,
                  const torch::Tensor& bias,
                  torch::Tensor& C,
                  int M, int N, int K)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    const float* bias_ptr = bias.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    gemm_forward_kernel<<<grid, block>>>(A_ptr, B_ptr, bias_ptr, C_ptr, M, N, K);
    cudaDeviceSynchronize();
}
"""

# --------------------------------------------------------------
# 2. C++ binding (pybind11)
# --------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void gemm_forward(const torch::Tensor& A,
                  const torch::Tensor& B,
                  const torch::Tensor& bias,
                  torch::Tensor& C,
                  int M, int N, int K);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_forward", &gemm_forward, "GEMM forward");
}
"""

# --------------------------------------------------------------
# 3. Compile the inline extension
# --------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# --------------------------------------------------------------
# 4. Helper that invokes the custom GEMM kernel
# --------------------------------------------------------------
def linear_cuda(x, weight, bias):
    """
    Custom fully-connected layer:  y = x @ Wᵀ + b
    Implemented with a hand-written CUDA GEMM kernel.
    """
    # Move tensors to GPU if needed and ensure row-major contiguous layout
    if not x.is_cuda:
        x = x.cuda()
    if not weight.is_cuda:
        weight = weight.cuda()
    if bias is not None and not bias.is_cuda:
        bias = bias.cuda()

    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    else:
        # No bias supplied → use a zero vector
        bias = torch.zeros(weight.size(0), dtype=x.dtype, device=x.device)

    M = x.size(0)        # batch size
    N = weight.size(0)   # out_features
    K = x.size(1)        # in_features

    # Allocate output buffer
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # Launch the fused GEMM + bias kernel
    fused_ext.gemm_forward(x, weight, bias, out, M, N, K)

    return out

# --------------------------------------------------------------
# 5. functional_model – the only function that will be imported
# --------------------------------------------------------------
def functional_model(
    x,
    y,
    *,
    bmm_weight,
    bmm_bias,
    instance_norm_running_mean,
    instance_norm_running_var,
    instance_norm_weight,
    instance_norm_bias,
    instance_norm_use_input_stats,
    instance_norm_momentum,
    instance_norm_eps,
):
    # ---- 1) Custom fused linear (GEMM + bias) ----
    x = linear_cuda(x, bmm_weight, bmm_bias)

    # ---- 2) Instance norm (unchanged) ----
    x = F.instance_norm(
        x.unsqueeze(1).unsqueeze(1),
        instance_norm_running_mean,
        instance_norm_running_var,
        instance_norm_weight,
        instance_norm_bias,
        use_input_stats=instance_norm_use_input_stats,
        momentum=instance_norm_momentum,
        eps=instance_norm_eps
    ).squeeze(1).squeeze(1)

    # ---- 3) Element-wise add & multiply ----
    x = x + y
    x = x * y

    return x

# --------------------------------------------------------------
# 6. Optional helper functions (not required by the evaluator)
# --------------------------------------------------------------
def get_init_inputs():
    in_features = 8192
    out_features = 8192
    return [in_features, out_features]

def get_inputs():
    batch_size = 1024
    in_features = 8192
    out_features = 8192
    return [torch.rand(batch_size, in_features), torch.rand(batch_size, out_features)]
