# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_144949/code_4.py
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

# The naive implementation of linear (MatMul + Bias) inside the kernel is inefficient 
# for large matrices. However, to guarantee no built-in matmul is used, we implemented 
# a tiled matrix multiplication kernel strategy for the linear layer.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void fused_op_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ norm_weight,
    const float* __restrict__ norm_bias,
    float* __restrict__ output,
    int batch_size, int in_features, int out_features,
    float eps
) {
    int batch_idx = blockIdx.y;
    int out_idx = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    // Shared memory for tiling input features to avoid repeated global loads
    __shared__ float s_x[TILE_SIZE];
    
    float val = bias[out_idx];
    for (int k = 0; k < (in_features + TILE_SIZE - 1) / TILE_SIZE; ++k) {
        int k_idx = k * TILE_SIZE + threadIdx.x;
        if (k_idx < in_features) {
            s_x[threadIdx.x] = x[batch_idx * in_features + k_idx];
        } else {
            s_x[threadIdx.x] = 0.0f;
        }
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; ++i) {
            if (k * TILE_SIZE + i < in_features) {
                val += s_x[i] * weight[out_idx * in_features + (k * TILE_SIZE + i)];
            }
        }
        __syncthreads();
    }
    
    // Instance Norm: For a single element vector (1x1xC), Instance Norm 
    // simplifies to (x-mean)/std. Since this is over 1 feature per instance,
    // the stats are trivial.
    float norm = (val - 0.0f) * rsqrtf(0.0f + eps) * norm_weight[out_idx] + norm_bias[out_idx];
    
    // Final element-wise fusion
    float y_val = y[batch_idx * out_features + out_idx];
    output[batch_idx * out_features + out_idx] = (norm + y_val) * y_val;
}

void fused_op_launcher(
    torch::Tensor x, torch::Tensor y, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor norm_w, torch::Tensor norm_b, torch::Tensor output,
    float eps
) {
    int batch = x.size(0);
    int out_f = weight.size(0);
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_f + TILE_SIZE - 1) / TILE_SIZE, batch);
    
    fused_op_kernel<<<grid, block>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.data_ptr<float>(), norm_w.data_ptr<float>(), norm_b.data_ptr<float>(), 
        output.data_ptr<float>(), batch, x.size(1), out_f, eps
    );
}
"""

cpp_source = "void fused_op_launcher(torch::Tensor x, torch::Tensor y, torch::Tensor weight, torch::Tensor bias, torch::Tensor norm_w, torch::Tensor norm_b, torch::Tensor output, float eps);"

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_op_launcher'],
    extra_cuda_cflags=['-O3']
)

def functional_model(
    x, y, *, bmm_weight, bmm_bias, instance_norm_running_mean, 
    instance_norm_running_var, instance_norm_weight, instance_norm_bias, 
    instance_norm_use_input_stats, instance_norm_momentum, instance_norm_eps
):
    output = torch.empty_like(y)
    fused_ext.fused_op_launcher(
        x, y, bmm_weight, bmm_bias, instance_norm_weight, 
        instance_norm_bias, output, instance_norm_eps
    )
    return output
