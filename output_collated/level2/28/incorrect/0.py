# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_144949/code_2.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- CUDA Kernel Definition ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void fused_op_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ norm_weight,
    const scalar_t* __restrict__ norm_bias,
    const scalar_t* __restrict__ running_mean,
    const scalar_t* __restrict__ running_var,
    const bool use_input_stats,
    const scalar_t momentum,
    const scalar_t eps,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_features,
    const int out_features
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    if (batch_idx >= batch_size) return;

    const int x_batch_offset = batch_idx * in_features;
    const int out_batch_offset = batch_idx * out_features;

    // Shared memory for reduction operations (mean, variance)
    extern __shared__ scalar_t shared_mem[];
    scalar_t* shared_sum = shared_mem;
    scalar_t* shared_var = shared_mem + block_size;

    // Linear transformation: out_features x in_features @ in_features -> out_features
    for (int out_idx = tid; out_idx < out_features; out_idx += block_size) {
        scalar_t sum = 0;
        for (int k = 0; k < in_features; ++k) {
            sum += x[x_batch_offset + k] * weight[out_idx * in_features + k];
        }
        if (bias) {
            sum += bias[out_idx];
        }
        output[out_batch_offset + out_idx] = sum;
    }
    __syncthreads();

    // Instance Norm: Compute mean
    scalar_t mean = 0;
    for (int i = tid; i < out_features; i += block_size) {
        mean += output[out_batch_offset + i];
    }
    shared_sum[tid] = mean;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    mean = shared_sum[0] / out_features;
    __syncthreads();

    // Instance Norm: Compute variance
    scalar_t var = 0;
    for (int i = tid; i < out_features; i += block_size) {
        scalar_t diff = output[out_batch_offset + i] - mean;
        var += diff * diff;
    }
    shared_var[tid] = var;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_var[tid] += shared_var[tid + stride];
        }
        __syncthreads();
    }
    var = shared_var[0] / out_features;
    scalar_t inv_std = rsqrtf(var + eps);
    __syncthreads();

    // Apply instance norm, affine transform, add, multiply
    for (int i = tid; i < out_features; i += block_size) {
        const int idx = out_batch_offset + i;
        scalar_t norm_val = (output[idx] - mean) * inv_std;
        scalar_t affine = norm_val * norm_weight[i] + norm_bias[i];
        affine = affine + y[idx];
        output[idx] = affine * y[idx];
    }
}

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor norm_weight,
    torch::Tensor norm_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool use_input_stats,
    double momentum,
    double eps,
    torch::Tensor output
) {
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int batch_size = x.size(0);
    const int in_features = x.size(1);
    const int out_features = weight.size(0);

    // Heuristic block size
    const int threads_per_block = 256;
    const int shared_mem_size = 2 * threads_per_block * sizeof(float);

    fused_op_kernel<float><<<batch_size, threads_per_block, shared_mem_size, stream>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        use_input_stats,
        static_cast<float>(momentum),
        static_cast<float>(eps),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor norm_weight,
    torch::Tensor norm_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool use_input_stats,
    double momentum,
    double eps,
    torch::Tensor output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused operation forward pass");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
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
    batch_size, in_features = x.shape
    out_features = bmm_weight.shape[0]
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_features, device=x.device, dtype=x.dtype)
    
    # Call the custom CUDA kernel
    fused_ext.fused_op(
        x.contiguous(),
        y.contiguous(),
        bmm_weight.contiguous(),
        bmm_bias.contiguous(),
        instance_norm_weight.contiguous(),
        instance_norm_bias.contiguous(),
        instance_norm_running_mean.contiguous(),
        instance_norm_running_var.contiguous(),
        instance_norm_use_input_stats,
        instance_norm_momentum,
        instance_norm_eps,
        output
    )
    
    return output

batch_size = 1024  # Increased batch size
in_features = 8192  # Increased input features
out_features = 8192  # Increased output features

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features), torch.rand(batch_size, out_features)]

