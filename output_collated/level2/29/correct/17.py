# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105411/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['linear_weight', 'linear_bias']
REQUIRED_FLAT_STATE_NAMES = ['linear_weight', 'linear_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Mish, and applies Mish again.
    """

    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

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

# --- CUDA Kernel ---
# We use a simplified tiling strategy fused with Mish activation in registers
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float mish(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

// Fusing Mish(Mish(x)) -> Mish(x) = x * tanh(softplus(x))
// This kernel performs the matmul + elementwise activation fusion
__global__ void fused_gemm_mish_kernel(const float* __restrict__ A, const float* __restrict__ B, 
                                       const float* __restrict__ bias, float* __restrict__ C, 
                                       int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[col * K + k]; // Simple dot product
        }
        acc += bias[col];
        
        // Fused Activation: Mish(Mish(x))
        float val = mish(mish(acc));
        
        C[row * N + col] = val;
    }
}

void fused_op_forward(int blocks_x, int blocks_y, int threads_x, int threads_y,
                      torch::Tensor A, torch::Tensor B, torch::Tensor bias, torch::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    dim3 threads(threads_x, threads_y);
    dim3 grid(blocks_x, blocks_y);
    fused_gemm_mish_kernel<<<grid, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), 
                                              bias.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function in the .cu file
void fused_op_forward(int blocks_x, int blocks_y, int threads_x, int threads_y,
                      torch::Tensor A, torch::Tensor B, torch::Tensor bias, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused GEMM + Double Mish");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_gemm_mish',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias):
    out = torch.empty((x.size(0), linear_weight.size(0)), device=x.device, dtype=x.dtype)
    # The weight is transposed to match CUDA kernel logic expected (N, K)
    M, K = x.shape
    N = linear_weight.shape[0]
    
    # Launch configuration
    threads_x, threads_y = 16, 16
    blocks_x = (N + threads_x - 1) // threads_x
    blocks_y = (M + threads_y - 1) // threads_y
    
    fused_ext.fused_op(blocks_x, blocks_y, threads_x, threads_y, x, linear_weight, linear_bias, out)
    return out

# Initialization logic for the evaluation
batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

# Note: Weights should be initialized and passed to functional_model externally via kwargs
def get_weights():
    w = torch.rand(out_features, in_features).cuda()
    b = torch.rand(out_features).cuda()
    return w, b
