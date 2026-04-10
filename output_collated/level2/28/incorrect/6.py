# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_145329/code_3.py
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
#  CUDA kernel (fused linear + instance‑norm + (x+y)*y)
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

 constexpr int I = 8192;   // input features
 constexpr int O = 8192;   // output features
 constexpr int BLOCK_DIM = 256;
 constexpr int FEAT_PER_THREAD = O / BLOCK_DIM;   // 32

 __global__ void fused_forward_kernel(
     const float* __restrict__ x,       // (B, I)
     const float* __restrict__ y,       // (B, O)
     const float* __restrict__ weight,  // (O, I)
     const float* __restrict__ bias,    // (O)
     const float* __restrict__ running_mean, // (O)
     const float* __restrict__ running_var, // (O)
     const float* __restrict__ inst_norm_weight, // (O)
     const float* __restrict__ inst_norm_bias,   // (O)
     int use_input_stats,
     float momentum,
     float eps,
     float* output,                     // (B, O)
     int B, int N_IN, int N_OUT)
 {
     // batch index handled by block id
     int batch_idx = blockIdx.x;
     if (batch_idx >= B) return;

     // ---- shared memory for the input vector of this batch element ----
     __shared__ float x_shared[I];   // 8192 floats

     // coalesced load of x into shared memory
     for (int i = threadIdx.x; i < N_IN; i += BLOCK_DIM) {
         x_shared[i] = x[batch_idx * N_IN + i];
     }
     __syncthreads();

     // ---- local storage for the linear results of the features assigned to this thread ----
     float local_out[FEAT_PER_THREAD];

     // ---- 1) Linear layer (F.linear) ----
     for (int k = 0; k < FEAT_PER_THREAD; ++k) {
         int f = threadIdx.x * FEAT_PER_THREAD + k;   // output feature index
         float sum = bias[f];
         const float* w_row = weight + f * N_IN;
         float dot = 0.0f;
         // dot product of x_shared (size I) with weight row
         for (int i = 0; i < N_IN; ++i) {
             dot += x_shared[i] * w_row[i];
         }
         local_out[k] = dot;
     }
     __syncthreads();

     // ---- 2) Per‑instance mean & variance (for instance norm) ----
     float sum_local = 0.0f;
     float sum_sq_local = 0.0f;
     for (int k = 0; k < FEAT_PER_THREAD; ++k) {
         float v = local_out[k];
         sum_local   += v;
         sum_sq_local += v * v;
     }

     __shared__ float s_sum[BLOCK_DIM];
     __shared__ float s_sum_sq[BLOCK_DIM];
     s_sum[threadIdx.x]   = sum_local;
     s_sum_sq[threadIdx.x] = sum_sq_local;
     __syncthreads();

     // parallel reduction inside the block
     for (unsigned int s = BLOCK_DIM / 2; s > 0; s >>= 1) {
         if (threadIdx.x < s) {
             s_sum[threadIdx.x]   += s_sum[threadIdx.x + s];
             s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
         }
         __syncthreads();
     }

     float mean = 0.0f, var = 0.0f;
     if (threadIdx.x == 0) {
         mean = s_sum[0] / N_OUT;
         var = (s_sum_sq[0] / N_OUT) - mean * mean;
         if (var < 0.0f) var = 0.0f;   // numerical safety
         var += eps;
     }
     __syncthreads();

     // ---- 3) Instance norm, 4) (norm + y) * y, write final output ----
     for (int k = 0; k < FEAT_PER_THREAD; ++k) {
         int f = threadIdx.x * FEAT_PER_THREAD + k;
         float val = local_out[k];

         // choose mean/var either from input (use_input_stats) or from running buffers
         float m, v;
         if (use_input_stats) {
             m = mean;
             v = var;
         } else {
             m = running_mean[f];
             v = running_var[f];
         }
         float std = sqrtf(v);
         float norm = (val - m) / std * inst_norm_weight[f] + inst_norm_bias[f];

         float y_val = y[batch_idx * N_OUT + f];
         float final_val = (norm + y_val) * y_val;

         output[batch_idx * N_OUT + f] = final_val;
     }
 }
 
 // -------------------------------------------------------------------------
 //  C++ interface that launches the kernel
 // -------------------------------------------------------------------------
 void fused_op(
     at::Tensor x,
     at::Tensor y,
     at::Tensor weight,
     at::Tensor bias,
     at::Tensor running_mean,
     at::Tensor running_var,
     at::Tensor inst_norm_weight,
     at::Tensor inst_norm_bias,
     bool use_input_stats,
     float momentum,
     float eps,
     at::Tensor output)
 {
     int B = x.size(0);
     int I = x.size(1);
     int O = weight.size(0);

     const float* x_ptr = x.data_ptr<float>();
     const float* y_ptr = y.data_ptr<float>();
     const float* w_ptr = weight.data_ptr<float>();
     const float* b_ptr = bias.data_ptr<float>();
     const float* rm_ptr = running_mean.data_ptr<float>();
     const float* rv_ptr = running_var.data_ptr<float>();
     const float* wn_ptr = inst_norm_weight.data_ptr<float>();
     const float* bn_ptr = inst_norm_bias.data_ptr<float>();
     float* out_ptr = output.data_ptr<float>();

     int use_input_stats_int = use_input_stats ? 1 : 0;

     const int blocks = B;          // one block per batch element
     const int threads = BLOCK_DIM; // 256 threads per block

     fused_forward_kernel<<<blocks, threads>>>(
         x_ptr, y_ptr, w_ptr, b_ptr,
         rm_ptr, rv_ptr, wn_ptr, bn_ptr,
         use_input_stats_int, momentum, eps,
         out_ptr, B, I, O);

     cudaDeviceSynchronize();
 }
 """

# -------------------------------------------------------------------------
#  PyBind11 binding
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op(
    at::Tensor x,
    at::Tensor y,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor running_mean,
    at::Tensor running_var,
    at::Tensor inst_norm_weight,
    at::Tensor inst_norm_bias,
    bool use_input_stats,
    float momentum,
    float eps,
    at::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused forward kernel");
}
"""

# -------------------------------------------------------------------------
#  Build the inline CUDA extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
#  Functional model used by the evaluator
# -------------------------------------------------------------------------
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
    # Ensure contiguous memory layout on the device
    x = x.contiguous()
    y = y.contiguous()
    bmm_weight = bmm_weight.contiguous()
    bmm_bias = bmm_bias.contiguous()
    instance_norm_running_mean = instance_norm_running_mean.contiguous()
    instance_norm_running_var = instance_norm_running_var.contiguous()
    instance_norm_weight = instance_norm_weight.contiguous()
    instance_norm_bias = instance_norm_bias.contiguous()

    # Output has the same shape as y (batch × out_features)
    output = torch.empty_like(y)

    # Launch the fused kernel
    fused_ext.fused_op(
        x, y,
        bmm_weight, bmm_bias,
        instance_norm_running_mean, instance_norm_running_var,
        instance_norm_weight, instance_norm_bias,
        instance_norm_use_input_stats,
        instance_norm_momentum,
        instance_norm_eps,
        output,
    )
    return output
