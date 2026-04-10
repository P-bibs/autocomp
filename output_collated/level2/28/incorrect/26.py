# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150941/code_3.py
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
from torch.utils.cpp_extension import load_inline

# CUDA kernel implementation
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tiled GEMM kernel: C = A @ B^T + bias
__global__ void gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int N, int M, int K) {
    
    const int BM = 16;
    const int BN = 16;
    const int BK = 16;
    
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    
    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int kk = 0; kk < K; kk += BK) {
        // Load A tile
        int a_col = kk + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < N && a_col < K) ? A[row * K + a_col] : 0.0f;
        
        // Load B tile
        int b_row = kk + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (col < M && b_row < K) ? B[col * K + b_row] : 0.0f;
        
        __syncthreads();
        
        // Compute partial sum
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result with bias
    if (row < N && col < M) {
        C[row * M + col] = sum + bias[col];
    }
}

// Fused instance norm + element-wise operations kernel
__global__ void fused_norm_kernel(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    bool use_input_stats,
    float eps,
    float* __restrict__ out,
    int N, int C) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) return;
    
    int c = idx % C;
    float x_val = X[idx];
    float y_val = Y[idx];
    
    float normalized;
    if (use_input_stats) {
        // For 1x1 spatial size, normalized value is 0
        normalized = 0.0f;
    } else {
        // Use running statistics
        float mean = running_mean[c];
        float var = running_var[c];
        normalized = (x_val - mean) * rsqrtf(var + eps);
    }
    
    // Apply scale and shift
    float scaled = normalized * weight[c] + bias[c];
    
    // Final element-wise operations: (scaled + y) * y
    out[idx] = (scaled + y_val) * y_val;
}

// Host wrapper functions
void gemm_forward(torch::Tensor A, torch::Tensor B, torch::Tensor bias,
                  torch::Tensor C, int N, int M, int K) {
    const int BM = 16, BN = 16;
    dim3 block(BM, BN);
    dim3 grid((M + BN - 1) / BN, (N + BM - 1) / BM);
    
    gemm_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        bias.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M, K
    );
    
    cudaDeviceSynchronize();
}

void fused_norm_forward(torch::Tensor X, torch::Tensor Y,
                        torch::Tensor running_mean, torch::Tensor running_var,
                        torch::Tensor weight, torch::Tensor bias,
                        bool use_input_stats, float eps,
                        torch::Tensor out, int N, int C) {
    const int blockSize = 256;
    int gridSize = (N * C + blockSize - 1) / blockSize;
    
    fused_norm_kernel<<<gridSize, blockSize>>>(
        X.data_ptr<float>(),
        Y.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        use_input_stats,
        eps,
        out.data_ptr<float>(),
        N, C
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void gemm_forward(torch::Tensor A, torch::Tensor B, torch::Tensor bias,
                  torch::Tensor C, int N, int M, int K);

void fused_norm_forward(torch::Tensor X, torch::Tensor Y,
                        torch::Tensor running_mean, torch::Tensor running_var,
                        torch::Tensor weight, torch::Tensor bias,
                        bool use_input_stats, float eps,
                        torch::Tensor out, int N, int C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm_forward, "Tiled GEMM kernel");
    m.def("fused_norm", &fused_norm_forward, "Fused instance norm and element-wise operations");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

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
    # Move tensors to GPU and ensure contiguous memory layout
    device = torch.device('cuda')
    x = x.to(device).contiguous()
    y = y.to(device).contiguous()
    bmm_weight = bmm_weight.to(device).contiguous()
    bmm_bias = bmm_bias.to(device).contiguous()
    running_mean = instance_norm_running_mean.to(device).contiguous()
    running_var = instance_norm_running_var.to(device).contiguous()
    weight = instance_norm_weight.to(device).contiguous()
    bias = instance_norm_bias.to(device).contiguous()
    
    batch_size = x.shape[0]
    in_features = x.shape[1]
    out_features = bmm_weight.shape[0]
    
    # Step 1: Custom GEMM operation
    gemm_output = torch.empty((batch_size, out_features), dtype=torch.float32, device=device)
    fused_ext.gemm(x, bmm_weight, bmm_bias, gemm_output, batch_size, out_features, in_features)
    
    # Step 2: Fused instance norm and element-wise operations
    final_output = torch.empty((batch_size, out_features), dtype=torch.float32, device=device)
    fused_ext.fused_norm(
        gemm_output, y,
        running_mean, running_var,
        weight, bias,
        instance_norm_use_input_stats,
        instance_norm_eps,
        final_output,
        batch_size, out_features
    )
    
    return final_output

# Constants for testing
batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features), torch.rand(batch_size, out_features)]
