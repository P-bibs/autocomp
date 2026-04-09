# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_014845/code_3.py
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

# -------------------------------------------------------------------------
#  CUDA source – fused kernel (linear + bias + ReLU)
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused kernel:  y = ReLU( x @ W^T + b )
//   x : (batch, in_features)  – row‑major
//   W : (out_features, in_features) – row‑major
//   b : (out_features,)
//   y : (batch, out_features)
__global__ void fused_linear_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float*       __restrict__ y,
    const int batch,
    const int in_feat,
    const int out_feat)
{
    // blockIdx.x  -> batch index
    // blockIdx.y  -> column tile index
    const int batch_idx = blockIdx.x;
    const int out_col   = blockIdx.y * blockDim.x + threadIdx.x;

    if (out_col >= out_feat) return;

    // --- accumulation in register ---------------------------------------
    float sum = 0.0f;

    // Shared memory tile for the input activations of the current batch
    __shared__ float tile_x[256];            // blockDim.x == 256

    // Loop over the inner dimension in tiles of size blockDim.x
    for (int i = 0; i < in_feat; i += blockDim.x) {
        // Each thread loads one element of the input tile
        const int idx = i + threadIdx.x;
        if (idx < in_feat) {
            tile_x[threadIdx.x] = x[batch_idx * in_feat + idx];
        } else {
            tile_x[threadIdx.x] = 0.0f;      // pad with zero
        }
        __syncthreads();

        // Each thread reads the weight element that corresponds to its output column
        if (idx < in_feat) {
            const float w_val = w[out_col * in_feat + idx];
            sum += tile_x[threadIdx.x] * w_val;
        }
        __syncthreads();
    }

    // --- bias + ReLU ----------------------------------------------------
    float out_val = sum + b[out_col];
    out_val = fmaxf(out_val, 0.0f);          // ReLU

    // Store result
    y[batch_idx * out_feat + out_col] = out_val;
}

// Wrapper that performs the kernel launch
void fused_op_forward(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor y)
{
    const int batch    = x.size(0);
    const int in_feat  = x.size(1);
    const int out_feat = w.size(0);

    const int threads = 256;                         // blockDim.x
    const int blocks_y = (out_feat + threads - 1) / threads;
    const dim3 blocks(batch, blocks_y);
    const dim3 threads_per_block(threads, 1, 1);

    fused_linear_relu_kernel<<<blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        y.data_ptr<float>(),
        batch, in_feat, out_feat);

    // Optional: cudaGetLastError() check could be added here
}
"""

# -------------------------------------------------------------------------
#  C++ binding – expose the fused kernel to Python via PyBind11
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the CUDA implementation
void fused_op_forward(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor y);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward,
          "Fused linear + bias + ReLU forward (CUDA)");
}
"""

# -------------------------------------------------------------------------
#  Build the inline extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
#  The functional model that will be evaluated
# -------------------------------------------------------------------------
def functional_model(
    x,
    *,
    gemm_weight,
    bias,
):
    """
    fused linear + bias + ReLU.
    All inputs are expected to be torch tensors on the GPU.
    """
    # Ensure the inputs are on the GPU and have the correct dtype (float32)
    if not x.is_cuda:
        x = x.cuda()
    if not gemm_weight.is_cuda:
        gemm_weight = gemm_weight.cuda()
    if not bias.is_cuda:
        bias = bias.cuda()

    # Allocate output tensor
    batch = x.size(0)
    out_features = gemm_weight.size(0)
    y = torch.empty((batch, out_features), dtype=x.dtype, device=x.device)

    # Call the fused CUDA kernel
    fused_ext.fused_op(x, gemm_weight, bias, y)

    return y

batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

def get_init_inputs():
    return [in_features, out_features, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
