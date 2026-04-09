# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_024220/code_8.py
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

# The strategy is to leverage cuBLAS for the heavy-lifting GEMM
# followed by a highly optimized, vectorized kernel for the element-wise 
# operations to minimize kernel launch overhead and memory latency.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Vectorized fused bias and ReLU kernel
// Using float4 for memory bandwidth maximization
__global__ void bias_relu_vectorized_kernel(float* x, const float* bias, int M, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int total_elements = M * N;

    if (idx < total_elements) {
        float4 val;
        // Load, bias add, and ReLU
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int current_idx = idx + i;
            if (current_idx < total_elements) {
                int col = current_idx % N;
                float v = x[current_idx] + bias[col];
                x[current_idx] = (v > 0.0f) ? v : 0.0f;
            }
        }
    }
}

void fused_linear_bias_relu(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);

    // 1. Perform GEMM using cuBLAS (Highly optimized for specific hardware)
    static cublasHandle_t handle = nullptr;
    if (!handle) cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // x: (M, K), weight: (N, K) -> Output: (M, N)
    // cuBLAS performs C = alpha*op(A)*op(B) + beta*C
    // op(weight) is N x K (transpose K x N), op(x) is M x K (transpose K x M)
    // To get (M, N), we do (x) * (weight)^T: (M, K) * (K, N)
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                N, M, K, 
                &alpha, weight.data_ptr<float>(), K, 
                x.data_ptr<float>(), K, 
                &beta, output.data_ptr<float>(), N);

    // 2. Optimized Fused Bias + ReLU kernel
    int num_elements = M * N;
    int threads = 256;
    int blocks = (num_elements / 4 + threads - 1) / threads;
    bias_relu_vectorized_kernel<<<blocks, threads>>>(output.data_ptr<float>(), bias.data_ptr<float>(), M, N);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_linear_bias_relu(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_bias_relu", &fused_linear_bias_relu, "Fused Linear Bias ReLU");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_ldflags=['-lcublas'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, gemm_weight, bias):
    """
    Optimized functional model using cuBLAS for GEMM and a 
    vectorized kernels for fused post-processing.
    """
    batch_size = x.size(0)
    out_features = bias.size(0)
    output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
    
    # Call the C++/CUDA extension
    fused_ext.fused_linear_bias_relu(x, gemm_weight, bias, output)
    return output
