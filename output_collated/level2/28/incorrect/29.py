# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150941/code_7.py
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

# -------------------------------------------------------------------------
# CUDA / C++ source – contains the optimized kernels for the entire pipeline
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Tiled GEMM kernel: C = A @ W^T + bias
// A: [N, K], W: [M, K], bias: [M], C: [N, M]
__global__ void gemm_kernel(const float* __restrict__ A,
                            const float* __restrict__ W,
                            const float* __restrict__ bias,
                            float* __restrict__ C,
                            int N, int M, int K) {
    const int BM = 32;
    const int BN = 32;
    const int BK = 16;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * BN + tx;
    int row = blockIdx.y * BM + ty;

    float sum = 0.0f;

    for (int kk = 0; kk < K; kk += BK) {
        if (row < N && (kk + tx) < K) As[ty][tx] = A[row * K + (kk + tx)];
        else As[ty][tx] = 0.0f;
        
        if (col < M && (kk + ty) < K) Bs[ty][tx] = W[col * K + (kk + ty)];
        else Bs[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < M) {
        C[row * M + col] = sum + bias[col];
    }
}

// Fuses InstanceNorm (1x1 spatial) + Add + Mul
__global__ void fused_norm_add_mul_kernel(const float* __restrict__ X,
                                          const float* __restrict__ Y,
                                          const float* __restrict__ mean,
                                          const float* __restrict__ var,
                                          const float* __restrict__ weight,
                                          const float* __restrict__ bias,
                                          int use_input_stats,
                                          float eps,
                                          float* __restrict__ out,
                                          int N, int C_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_dim) return;

    int c = idx % C_dim;
    float x_val = X[idx];
    float y_val = Y[idx];
    
    float norm = 0.0f;
    if (!use_input_stats) {
        norm = (x_val - mean[c]) * rsqrtf(var[c] + eps);
    }
    
    float after_norm = norm * weight[c] + bias[c];
    out[idx] = (after_norm + y_val) * y_val;
}

void call_kernels(torch::Tensor A, torch::Tensor W, torch::Tensor b,
                  torch::Tensor Y, torch::Tensor rm, torch::Tensor rv,
                  torch::Tensor iw, torch::Tensor ib,
                  torch::Tensor out, int use_input_stats, float eps) {
    int N = A.size(0);
    int K = A.size(1);
    int M = W.size(0);
    
    auto tmp = torch::empty({N, M}, A.options());
    
    dim3 gemm_block(32, 32);
    dim3 gemm_grid((M + 31) / 32, (N + 31) / 32);
    gemm_kernel<<<gemm_grid, gemm_block>>>(
        A.data_ptr<float>(), W.data_ptr<float>(), b.data_ptr<float>(), 
        tmp.data_ptr<float>(), N, M, K
    );
    
    int total = N * M;
    int block2 = 256;
    int grid2 = (total + block2 - 1) / block2;
    fused_norm_add_mul_kernel<<<grid2, block2>>>(
        tmp.data_ptr<float>(), Y.data_ptr<float>(), rm.data_ptr<float>(),
        rv.data_ptr<float>(), iw.data_ptr<float>(), ib.data_ptr<float>(),
        use_input_stats, eps, out.data_ptr<float>(), N, M
    );
}
"""

cpp_source = """
void call_kernels(torch::Tensor A, torch::Tensor W, torch::Tensor b,
                  torch::Tensor Y, torch::Tensor rm, torch::Tensor rv,
                  torch::Tensor iw, torch::Tensor ib,
                  torch::Tensor out, int use_input_stats, float eps);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &call_kernels, "Launch gemm and norm fusion");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, y, *, bmm_weight, bmm_bias, instance_norm_running_mean,
                     instance_norm_running_var, instance_norm_weight, instance_norm_bias,
                     instance_norm_use_input_stats, instance_norm_momentum, instance_norm_eps):
    x, y = x.cuda().contiguous(), y.cuda().contiguous()
    b_w, b_b = bmm_weight.cuda().contiguous(), bmm_bias.cuda().contiguous()
    rm, rv = instance_norm_running_mean.cuda().contiguous(), instance_norm_running_var.cuda().contiguous()
    iw, ib = instance_norm_weight.cuda().contiguous(), instance_norm_bias.cuda().contiguous()
    
    out = torch.empty_like(y)
    fused_ext.launch(x, b_w, b_b, y, rm, rv, iw, ib, out, int(instance_norm_use_input_stats), float(instance_norm_eps))
    return out
