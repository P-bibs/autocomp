# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_145711/code_4.py
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

# The linear operation (matrix-vector multiplication) is the bottleneck.
# We implement a tiled matmul kernel to ensure high occupancy and performance.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void fused_op_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ norm_weight,
    const float* __restrict__ norm_bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    float eps
) {
    // Shared memory for tiling weight matrix
    __shared__ float s_weight[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Linear transformation accumulator (per thread)
    float acc = 0.0f;
    for (int t = 0; t < (in_features + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < out_features && (t * TILE_SIZE + threadIdx.x) < in_features)
            s_weight[threadIdx.y][threadIdx.x] = weight[row * in_features + (t * TILE_SIZE + threadIdx.x)];
        else
            s_weight[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            int in_idx = t * TILE_SIZE + k;
            if (in_idx < in_features) {
                acc += x[blockIdx.z * in_features + in_idx] * s_weight[threadIdx.y][k];
            }
        }
        __syncthreads();
    }

    if (row < out_features) {
        float linear_res = acc + (bias ? bias[row] : 0.0f);
        
        // Instance Norm (using provided running stats)
        float inv_std = rsqrtf(running_var[row] + eps);
        float norm_res = (linear_res - running_mean[row]) * inv_std;
        
        if (norm_weight) norm_res *= norm_weight[row];
        if (norm_bias) norm_res += norm_bias[row];
        
        // Final op: (norm + y) * y
        float y_val = y[blockIdx.z * out_features + row];
        output[blockIdx.z * out_features + row] = (norm_res + y_val) * y_val;
    }
}

torch::Tensor fused_op(
    torch::Tensor x, torch::Tensor y, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, 
    torch::Tensor norm_weight, torch::Tensor norm_bias, float eps
) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);
    auto output = torch::empty_like(y);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_features + TILE_SIZE - 1) / TILE_SIZE, 1, batch_size);

    fused_op_kernel<<<grid, block>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.data_ptr<float>(), running_mean.data_ptr<float>(), 
        running_var.data_ptr<float>(), norm_weight.data_ptr<float>(), 
        norm_bias.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_features, out_features, eps
    );
    return output;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor fused_op(torch::Tensor x, torch::Tensor y, torch::Tensor weight, torch::Tensor bias,
                       torch::Tensor running_mean, torch::Tensor running_var, 
                       torch::Tensor norm_weight, torch::Tensor norm_bias, float eps);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused linear+norm+math kernel");
}
"""

fused_ext = load_inline(name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
                       extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, y, *, bmm_weight, bmm_bias, instance_norm_running_mean, 
                     instance_norm_running_var, instance_norm_weight, instance_norm_bias, 
                     instance_norm_use_input_stats, instance_norm_momentum, instance_norm_eps):
    return fused_ext.fused_op(
        x, y, bmm_weight, bmm_bias, 
        instance_norm_running_mean, instance_norm_running_var, 
        instance_norm_weight, instance_norm_bias, instance_norm_eps
    )
