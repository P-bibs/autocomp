# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_121229/code_14.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'bias_shape', 'stride', 'padding', 'output_padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
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
    # State for conv_transpose (nn.ConvTranspose2d)
    if 'conv_transpose_weight' in flat_state:
        state_kwargs['conv_transpose_weight'] = flat_state['conv_transpose_weight']
    else:
        state_kwargs['conv_transpose_weight'] = getattr(model.conv_transpose, 'weight', None)
    if 'conv_transpose_bias' in flat_state:
        state_kwargs['conv_transpose_bias'] = flat_state['conv_transpose_bias']
    else:
        state_kwargs['conv_transpose_bias'] = getattr(model.conv_transpose, 'bias', None)
    state_kwargs['conv_transpose_stride'] = model.conv_transpose.stride
    state_kwargs['conv_transpose_padding'] = model.conv_transpose.padding
    state_kwargs['conv_transpose_output_padding'] = model.conv_transpose.output_padding
    state_kwargs['conv_transpose_groups'] = model.conv_transpose.groups
    state_kwargs['conv_transpose_dilation'] = model.conv_transpose.dilation
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
# Custom CUDA kernel: bias subtraction + tanh, using shared memory for bias
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_op_kernel(
    float* __restrict__ data,
    const float* __restrict__ bias,
    const int N,
    const int C,
    const int HW)
{
    // Dynamic shared memory for the bias vector (size = C floats)
    extern __shared__ float bias_shared[];

    const int tid = threadIdx.x;

    // ---- Load bias into shared memory ------------------------------------
    if (tid < C) {
        bias_shared[tid] = __ldg(&bias[tid]);
    }
    __syncthreads();

    // ---- Grid‑strided loop to handle any tensor size --------------------
    int idx = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    const int total = N * C * HW;

    for (int i = idx; i < total; i += stride) {
        // Recover N, C, HW indices (same ordering as original kernel)
        int n = i / (C * HW);
        int c = (i / HW) % C;
        // int hw = i % HW;   // not needed for the compute

        // Read input, subtract bias (from shared memory), apply tanh
        float val = __ldg(&data[i]);
        val = tanhf(val - bias_shared[c]);
        data[i] = val;
    }
}

void fused_op_forward(torch::Tensor& x, const torch::Tensor& bias) {
    const int N = x.size(0);
    const int C = x.size(1);
    const int HW = x.size(2) * x.size(3);
    const int total = N * C * HW;

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    // Shared memory size = C floats
    const int shared_mem = C * static_cast<int>(sizeof(float));

    fused_op_kernel<<<blocks, threads, shared_mem>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, HW
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11) to expose the kernel to Python
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor& x, const torch::Tensor& bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward,
          "Fused bias subtraction + tanh (shared‑memory version)");
}
"""

# Compile the inline extension
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional model: transposed convolution + fused bias/tanh
# -------------------------------------------------------------------------
def functional_model(
    x,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    bias,
):
    # 1) Transposed convolution (PyTorch backend)
    x = torch.conv_transpose2d(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        stride=conv_transpose_stride,
        padding=conv_transpose_padding,
        output_padding=conv_transpose_output_padding,
        groups=conv_transpose_groups,
        dilation=conv_transpose_dilation,
    )

    # 2) Flatten bias to a 1‑D tensor (the kernel expects a 1‑D view)
    bias_flat = bias.view(-1)

    # 3) Custom fused kernel: (x - bias) and tanh in one pass,
    #    using shared memory for the bias vector.
    fused_ext.fused_op_forward(x, bias_flat)

    return x

# -------------------------------------------------------------------------
# Test scaffolding (not required for the submission, but kept for sanity)
# -------------------------------------------------------------------------
batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]
