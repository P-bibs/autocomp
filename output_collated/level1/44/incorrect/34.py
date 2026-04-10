# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_115947/code_6.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['kernel_size', 'stride', 'padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['avg_pool_kernel_size', 'avg_pool_stride', 'avg_pool_padding', 'avg_pool_ceil_mode', 'avg_pool_count_include_pad']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs 1D Average Pooling.
    """

    def __init__(self, kernel_size: int, stride: int=1, padding: int=0):
        """
        Initializes the 1D Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

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
    # State for avg_pool (nn.AvgPool1d)
    state_kwargs['avg_pool_kernel_size'] = model.avg_pool.kernel_size
    state_kwargs['avg_pool_stride'] = model.avg_pool.stride
    state_kwargs['avg_pool_padding'] = model.avg_pool.padding
    state_kwargs['avg_pool_ceil_mode'] = model.avg_pool.ceil_mode
    state_kwargs['avg_pool_count_include_pad'] = model.avg_pool.count_include_pad
    missing = [name for name in REQUIRED_STATE_NAMES if name not in state_kwargs]
    if missing:
        raise RuntimeError(f'Missing required state entries: {missing}')
    return state_kwargs


def get_functional_inputs():
    model = build_reference_model()
    forward_args = tuple(get_inputs())
    state_kwargs = extract_state_kwargs(model)
    return forward_args, state_kwargs




# ==============================================================
# avg_pool_fused.py
# --------------------------------------------------------------
# Implements functional_model using a fused CUDA kernel for
# high-performance average pooling.
# ==============================================================
import torch
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# CUDA kernel: fused average-pooling 1-D
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

template <typename scalar_t>
__global__ void avg_pool1d_fwd_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t N,
    const int64_t C,
    const int64_t L_in,
    const int64_t L_out,
    const int64_t kernel,
    const int64_t stride,
    const int64_t padding,
    const bool count_include_pad)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = N * C * L_out;

    for (; idx < total; idx += blockDim.x * gridDim.x) {
        const int64_t out_x = idx % L_out;
        const int64_t tmp   = idx / L_out;
        const int64_t c     = tmp % C;
        const int64_t n     = tmp / C;

        int64_t start = out_x * stride - padding;
        int64_t end   = start + kernel;

        int64_t real_start = max(start, (int64_t)0);
        int64_t real_end   = min(end,   L_in);

        scalar_t sum = 0;
        for (int64_t i = real_start; i < real_end; ++i) {
            sum += input[n * (C * L_in) + c * L_in + i];
        }

        scalar_t divisor = count_include_pad ? (scalar_t)kernel : (scalar_t)(real_end - real_start);
        output[n * (C * L_out) + c * L_out + out_x] = sum / divisor;
    }
}

void avg_pool1d_fwd(
    torch::Tensor input,
    torch::Tensor output,
    const int64_t kernel,
    const int64_t stride,
    const int64_t padding,
    const bool ceil_mode,
    const bool count_include_pad)
{
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t L_out = output.size(2);
    const int64_t L_in = input.size(2);

    const int threads = 256;
    const int blocks = (N * C * L_out + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "avg_pool1d_fwd", ([&] {
        avg_pool1d_fwd_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, C, L_in, L_out,
            kernel, stride, padding,
            count_include_pad);
    }));
}
"""

cpp_source = r"""
#include <torch/extension.h>
void avg_pool1d_fwd(torch::Tensor input, torch::Tensor output, const int64_t kernel, const int64_t stride, const int64_t padding, const bool ceil_mode, const bool count_include_pad);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_fwd", &avg_pool1d_fwd, "Fused average pool 1D");
}
"""

# Compile extension
_fused = load_inline(
    name="fused_avg_pool",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    avg_pool_kernel_size,
    avg_pool_stride,
    avg_pool_padding,
    avg_pool_ceil_mode,
    avg_pool_count_include_pad,
):
    N, C, L_in = x.shape
    if avg_pool_ceil_mode:
        L_out = (L_in + 2 * avg_pool_padding - avg_pool_kernel_size + avg_pool_stride - 1) // avg_pool_stride + 1
    else:
        L_out = (L_in + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1
        
    out = torch.empty((N, C, L_out), device=x.device, dtype=x.dtype)
    _fused.avg_pool1d_fwd(x, out, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding, avg_pool_ceil_mode, avg_pool_count_include_pad)
    return out

# Global config variables
batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4

def get_init_inputs():
    return [kernel_size, stride, padding]

def get_inputs():
    return [torch.rand(batch_size, in_channels, input_length, device="cuda")]
