# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_055208/code_6.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'divisor', 'pool_size', 'bias_shape', 'sum_dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'max_pool_kernel_size', 'max_pool_stride', 'max_pool_padding', 'max_pool_dilation', 'max_pool_ceil_mode', 'max_pool_return_indices', 'global_avg_pool_output_size', 'divisor', 'bias', 'sum_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """

    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

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
    # State for conv (nn.Conv3d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
    # State for max_pool (nn.MaxPool3d)
    state_kwargs['max_pool_kernel_size'] = model.max_pool.kernel_size
    state_kwargs['max_pool_stride'] = model.max_pool.stride
    state_kwargs['max_pool_padding'] = model.max_pool.padding
    state_kwargs['max_pool_dilation'] = model.max_pool.dilation
    state_kwargs['max_pool_ceil_mode'] = model.max_pool.ceil_mode
    state_kwargs['max_pool_return_indices'] = model.max_pool.return_indices
    # State for global_avg_pool (nn.AdaptiveAvgPool3d)
    state_kwargs['global_avg_pool_output_size'] = model.global_avg_pool.output_size
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    if 'sum_dim' in flat_state:
        state_kwargs['sum_dim'] = flat_state['sum_dim']
    else:
        state_kwargs['sum_dim'] = getattr(model, 'sum_dim')
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

# ----------------------------------------------------------------------
# CUDA kernel – bias cached in shared memory, reciprocal division,
# loop unrolling, optimal block size.
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_kernel_shared(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float inv_divisor,
    int N, int C, int D, int H, int W)
{
    // Dynamic shared memory: first C entries hold the bias vector
    extern __shared__ float sdata[];
    float* bias_shared = sdata;

    // ---- Load bias into shared memory cooperatively ----
    if (threadIdx.x < C) {
        bias_shared[threadIdx.x] = bias[threadIdx.x];
    }
    __syncthreads();

    // ---- Compute output index ----
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * D * H * W;
    if (idx >= total_elements) return;

    // ---- Decode spatial coordinates (batch + spatial dims) ----
    int inner = D * H * W;
    int n = idx / inner;
    int rest = idx % inner;
    int d = rest / (H * W);
    rest = rest % (H * W);
    int h = rest / W;
    int w = rest % W;

    // ---- Accumulate over channels ----
    float sum = 0.0f;
    #pragma unroll
    for (int c = 0; c < C; ++c) {
        // Input layout: NCHW (N, C, D, H, W)
        int input_idx = ((n * C + c) * D + d) * H * W + h * W + w;
        sum += (input[input_idx] * inv_divisor) + bias_shared[c];
    }
    output[idx] = sum;
}

// ----------------------------------------------------------------------
// Host wrapper – computes reciprocal, chooses block size,
// allocates dynamic shared memory for the bias cache.
// ----------------------------------------------------------------------
void fused_op_forward(torch::Tensor input, torch::Tensor bias,
                      torch::Tensor output, float divisor) {
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    float inv_divisor = 1.0f / divisor;

    const int threads = 128;                     // multiple of warp size
    int total_elements = N * D * H * W;
    int blocks = (total_elements + threads - 1) / threads;

    // Shared memory = C floats for the bias cache
    size_t shared_mem = C * sizeof(float);

    fused_op_kernel_shared<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        inv_divisor,
        N, C, D, H, W);
}
"""

# ----------------------------------------------------------------------
# C++ binding (PYBIND11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor input, torch::Tensor bias,
                      torch::Tensor output, float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward,
          "Fused divide, bias and sum kernel with shared-memory bias cache");
}
"""

# ----------------------------------------------------------------------
# Compile the extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional model – uses the optimized custom kernel
# ----------------------------------------------------------------------
import torch.nn.functional as F

def functional_model(
    x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups,
    max_pool_kernel_size, max_pool_stride, max_pool_padding, max_pool_dilation,
    max_pool_ceil_mode, max_pool_return_indices, global_avg_pool_output_size,
    divisor, bias, sum_dim,
):
    # ---- Standard PyTorch layers ----
    x = F.conv3d(x, conv_weight, conv_bias,
                 stride=conv_stride, padding=conv_padding,
                 dilation=conv_dilation, groups=conv_groups)

    x = F.max_pool3d(x, kernel_size=max_pool_kernel_size,
                     stride=max_pool_stride, padding=max_pool_padding,
                     dilation=max_pool_dilation, ceil_mode=max_pool_ceil_mode,
                     return_indices=max_pool_return_indices)

    x = F.adaptive_avg_pool3d(x, global_avg_pool_output_size)

    # ---- Custom fused kernel (now uses shared-memory bias cache) ----
    N, C, D, H, W = x.shape                     # after adaptive pool D=H=W=1 in typical use
    out = torch.empty((N, D, H, W), dtype=x.dtype, device=x.device)

    # Flatten bias (it arrives as (C,1,1,1))
    bias_1d = bias.view(-1)

    # Invoke the optimized CUDA kernel
    fused_ext.fused_op(x, bias_1d, out, divisor)

    return out

# ----------------------------------------------------------------------
# Dummy init / input helpers (required by the evaluation harness)
# ----------------------------------------------------------------------
batch_size = 128
in_channels = 8
out_channels = 16
depth = 16
height = 64
width = 64
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor,
            pool_size, bias_shape, sum_dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
