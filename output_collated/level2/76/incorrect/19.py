# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_022315/code_14.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'bias_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['gemm_weight', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['gemm_weight', 'bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a bias term, and applies ReLU.
    """

    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

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
    # State for gemm (nn.Linear)
    if 'gemm_weight' in flat_state:
        state_kwargs['gemm_weight'] = flat_state['gemm_weight']
    else:
        state_kwargs['gemm_weight'] = getattr(model.gemm, 'weight', None)
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Using 32x8 block size for tiling
#define BM 8
#define BN 32
#define BK 32

__global__ void fused_gemm_bias_relu_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    const float* __restrict__ bias,
    float* __restrict__ C, 
    int M, int N, int K) 
{
    __shared__ float Ashared[BM * BK];
    __shared__ float Bshared[BK * BN];

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    float sum = 0.0f;

    for (int k = 0; k < K; k += BK) {
        // Coalesced loads into shared memory
        // Load A tile: BM x BK
        for (int i = ty; i < BM; i += blockDim.y) {
            for (int j = tx; j < BK; j += blockDim.x) {
                int row = blockRow * BM + i;
                int col = k + j;
                Ashared[i * BK + j] = (row < M && col < K) ? A[row * K + col] : 0.0f;
            }
        }
        // Load B tile: BK x BN
        for (int i = ty; i < BK; i += blockDim.y) {
            for (int j = tx; j < BN; j += blockDim.x) {
                int row = k + i;
                int col = blockCol * BN + j;
                Bshared[i * BN + j] = (row < K && col < N) ? B[row * N + col] : 0.0f;
            }
        }
        __syncthreads();

        // Compute dot product
        #pragma unroll
        for (int e = 0; e < BK; ++e) {
            sum += Ashared[ty * BK + e] * Bshared[e * BN + tx];
        }
        __syncthreads();
    }

    // Write final result with fused bias and relu
    int row = blockRow * BM + ty;
    int col = blockCol * BN + tx;
    if (row < M && col < N) {
        float val = sum + bias[col];
        C[row * N + col] = (val > 0.0f) ? val : 0.0f;
    }
}

void launch_fused_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor bias, torch::Tensor C) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    dim3 threads(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    fused_gemm_bias_relu_kernel<<<grid, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), bias.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K
    );
}
"""

cpp_source = r"""
void launch_fused_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor bias, torch::Tensor C);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gemm", &launch_fused_gemm, "Fused GEMM Bias ReLU");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, gemm_weight, bias):
    """
    Computes (x @ gemm_weight.T) + bias with ReLU activation.
    gemm_weight is expected to be (out_features, in_features).
    """
    batch_size = x.size(0)
    out_features = bias.size(0)
    output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
    
    # gemm_weight orientation: 
    # The kernel expects B: (BK_dim x N) for access. 
    # Input gemm_weight is (N, K), which matches the kernel's B matrix.
    fused_ext.fused_gemm(x, gemm_weight, bias, output)
    return output
