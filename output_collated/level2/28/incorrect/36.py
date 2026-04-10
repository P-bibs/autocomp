# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_151636/code_0.py
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

# --- CUDA Kernel Code ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// Fused kernel performing:
// 1. Linear (MatMul + Bias)
// 2. InstanceNorm (with fused stats)
// 3. Residual Add (x + y)
// 4. Elementwise Multiply (x * y)
__global__ void fused_model_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,       // [out_features, in_features]
    const float* __restrict__ bias,         // [out_features]
    const float* __restrict__ norm_weight,  // [out_features]
    const float* __restrict__ norm_bias,    // [out_features]
    const float momentum,
    const float eps,
    float* __restrict__ out,
    float* __restrict__ running_mean,       // [out_features]
    float* __restrict__ running_var,        // [out_features]
    const int batch_size,
    const int in_features,
    const int out_features,
    const bool use_input_stats
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    const int N = batch_size * out_features;

    for (int idx = tid; idx < N; idx += total_threads) {
        const int b = idx / out_features;
        const int c = idx % out_features;

        // --- Step 1: Linear (MatMul + Bias) ---
        float val = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            val += x[b * in_features + k] * weight[c * in_features + k];
        }
        val += bias[c];

        // --- Step 2: InstanceNorm (per-channel stats) ---
        float mean = 0.0f, var = 0.0f;
        if (use_input_stats) {
            // For this simplified case, we assume spatial dimension is 1
            // So mean = val and var = 0 (but avoid degenerate case)
            mean = val;
            var = 0.0f;
        } else {
            mean = running_mean[c];
            var = running_var[c];
        }

        float inv_std = rsqrtf(var + eps);
        float norm_val = (val - mean) * inv_std;
        float out_val = norm_weight[c] * norm_val + norm_bias[c];

        // Update running stats (only one thread per channel)
        if (use_input_stats && (idx % out_features) == 0) {
            float old_mean = running_mean[c];
            float old_var = running_var[c];
            float new_mean = old_mean * (1.0f - momentum) + mean * momentum;
            float new_var = old_var * (1.0f - momentum) + var * momentum;
            running_mean[c] = new_mean;
            running_var[c] = new_var;
        }

        // --- Step 3: Residual Add & Multiply ---
        float y_val = y[idx];
        out_val = (out_val + y_val) * y_val;

        // Store final output
        out[idx] = out_val;
    }
}

void fused_model_forward(
    const at::Tensor &x,
    const at::Tensor &y,
    const at::Tensor &weight,
    const at::Tensor &bias,
    const at::Tensor &norm_weight,
    const at::Tensor &norm_bias,
    const float momentum,
    const float eps,
    at::Tensor &out,
    at::Tensor &running_mean,
    at::Tensor &running_var,
    const int batch_size,
    const int in_features,
    const int out_features,
    const bool use_input_stats
) {
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(norm_weight);
    CHECK_INPUT(norm_bias);
    CHECK_INPUT(running_mean);
    CHECK_INPUT(running_var);
    CHECK_INPUT(out);

    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int N = batch_size * out_features;
    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    fused_model_kernel<<<blocks, threads_per_block, 0, stream>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        momentum,
        eps,
        out.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        use_input_stats
    );
}
"""

# --- C++ Bindings ---
cpp_source = r"""
#include <torch/extension.h>

void fused_model_forward(
    const at::Tensor &x,
    const at::Tensor &y,
    const at::Tensor &weight,
    const at::Tensor &bias,
    const at::Tensor &norm_weight,
    const at::Tensor &norm_bias,
    const float momentum,
    const float eps,
    at::Tensor &out,
    at::Tensor &running_mean,
    at::Tensor &running_var,
    const int batch_size,
    const int in_features,
    const int out_features,
    const bool use_input_stats
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_model_forward", &fused_model_forward, "Fused Functional Model Forward");
}
"""

# --- Compile CUDA Extension ---
fused_ext = load_inline(
    name='fused_model_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# --- Reimplemented functional_model using fused CUDA kernel ---
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
    batch_size, in_features = x.shape
    out_features = y.shape[1]
    
    # Ensure all tensors are contiguous and on the same device
    x = x.contiguous()
    y = y.contiguous()
    bmm_weight = bmm_weight.contiguous()
    bmm_bias = bmm_bias.contiguous()
    instance_norm_weight = instance_norm_weight.contiguous()
    instance_norm_bias = instance_norm_bias.contiguous()
    instance_norm_running_mean = instance_norm_running_mean.contiguous()
    instance_norm_running_var = instance_norm_running_var.contiguous()

    # Pre-allocate output
    out = torch.empty_like(y)

    # Launch fused kernel
    fused_ext.fused_model_forward(
        x, y,
        bmm_weight, bmm_bias,
        instance_norm_weight, instance_norm_bias,
        instance_norm_momentum,
        instance_norm_eps,
        out,
        instance_norm_running_mean,
        instance_norm_running_var,
        batch_size, in_features, out_features,
        instance_norm_use_input_stats
    )

    return out

# --- Setup functions for testing ---
batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda'), torch.rand(batch_size, out_features, device='cuda')]
