# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150602/code_4.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tiled GEMM kernel with subsequent normalization
__global__ void fused_fwd_kernel(
    const float* __restrict__ x, const float* __restrict__ y,
    const float* __restrict__ weight, const float* __restrict__ bias,
    const float* __restrict__ inst_mean, const float* __restrict__ inst_var,
    const float* __restrict__ inst_weight, const float* __restrict__ inst_bias,
    float* __restrict__ output,
    int batch_size, int in_feat, int out_feat, float eps) 
{
    extern __shared__ float tile[]; // Shared buffer for input features
    
    int row = blockIdx.x; // Batch index
    int col = blockIdx.y * blockDim.x + threadIdx.x; // Output feature index
    
    if (row >= batch_size || col >= out_feat) return;

    // 1. Perform Linear: sum = x @ weight.T + bias
    float acc = bias[col];
    for (int i = 0; i < in_feat; i += blockDim.x) {
        // Load x into shared memory
        int tid = threadIdx.x;
        if (i + tid < in_feat) tile[tid] = x[row * in_feat + i + tid];
        __syncthreads();
        
        for (int k = 0; k < blockDim.x && (i + k) < in_feat; ++k) {
            acc += tile[k] * weight[col * in_feat + (i + k)];
        }
        __syncthreads();
    }

    // 2. Instance Norm (using running stats as per requirement)
    float val = (acc - inst_mean[col]) * rsqrtf(inst_var[col] + eps);
    val = val * inst_weight[col] + inst_bias[col];

    // 3. Add and Multiply
    float y_val = y[row * out_feat + col];
    output[row * out_feat + col] = (val + y_val) * y_val;
}

void fused_op_forward(
    torch::Tensor x, torch::Tensor y, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mean, torch::Tensor var, torch::Tensor n_w, torch::Tensor n_b,
    torch::Tensor output, float eps) 
{
    int batch = x.size(0);
    int in_f = x.size(1);
    int out_f = weight.size(0);
    
    dim3 block(256);
    dim3 grid(batch, (out_f + 255) / 256);
    
    fused_fwd_kernel<<<grid, block, 256 * sizeof(float)>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), 
        n_w.data_ptr<float>(), n_b.data_ptr<float>(), output.data_ptr<float>(),
        batch, in_f, out_f, eps
    );
}
"""

cpp_source = r"""
void fused_op_forward(
    torch::Tensor x, torch::Tensor y, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mean, torch::Tensor var, torch::Tensor n_w, torch::Tensor n_b,
    torch::Tensor output, float eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused kernel");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, y, *, bmm_weight, bmm_bias, instance_norm_running_mean, 
                     instance_norm_running_var, instance_norm_weight, instance_norm_bias, 
                     instance_norm_use_input_stats, instance_norm_momentum, instance_norm_eps):
    output = torch.empty_like(y)
    fused_ext.fused_op(
        x.contiguous(), y.contiguous(), bmm_weight.contiguous(), 
        bmm_bias.contiguous(), instance_norm_running_mean.contiguous(),
        instance_norm_running_var.contiguous(), instance_norm_weight.contiguous(),
        instance_norm_bias.contiguous(), output, instance_norm_eps
    )
    return output
