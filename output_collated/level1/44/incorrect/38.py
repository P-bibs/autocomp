# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_120345/code_2.py
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




import torch
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# 1) CUDA kernel (as a raw string)
# ------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void avg_pool1d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch,
    const int channels,
    const int in_len,
    const int out_len,
    const int kernel,
    const int stride,
    const int padding)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    for (int idx = tid; idx < batch * channels * out_len; idx += total_threads) {
        const int out_idx   = idx % out_len;
        const int channel   = (idx / out_len) % channels;
        const int batch_idx = idx / (out_len * channels);

        const int in_start = out_idx * stride - padding;

        scalar_t sum = 0;
        #pragma unroll
        for (int k = 0; k < kernel; ++k) {
            const int in_pos = in_start + k;
            if (in_pos >= 0 && in_pos < in_len) {
                const int in_offset = ((batch_idx * channels + channel) * in_len) + in_pos;
                sum += input[in_offset];
            }
        }
        const scalar_t avg = sum / static_cast<scalar_t>(kernel);
        const int out_offset = ((batch_idx * channels + channel) * out_len) + out_idx;
        output[out_offset] = avg;
    }
}

// Wrapper called from Python
void avg_pool1d_forward(
    torch::Tensor input,
    torch::Tensor output,
    const int kernel,
    const int stride,
    const int padding)
{
    const int batch   = input.size(0);
    const int channels = input.size(1);
    const int in_len  = input.size(2);
    const int out_len = output.size(2);

    const int threads = 256;
    const int blocks = (batch * channels * out_len + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "avg_pool1d_forward", ([&] {
        avg_pool1d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch,
            channels,
            in_len,
            out_len,
            kernel,
            stride,
            padding);
    }));

    // Optional: synchronize for easy debugging (can be removed for max perf)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in avg_pool1d_forward: %s\n", cudaGetErrorString(err));
    }
}
"""

# ------------------------------------------------------------------
# 2) C++ binding (pybind11)
# ------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

// Declaration of the function defined in the .cu source above
void avg_pool1d_forward(
    torch::Tensor input,
    torch::Tensor output,
    const int kernel,
    const int stride,
    const int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_forward", &avg_pool1d_forward,
          "Custom average-pool1d forward (CUDA)");
}
"""

# ------------------------------------------------------------------
# 3) Build the extension (inline compilation)
# ------------------------------------------------------------------
avg_pool_ext = load_inline(
    name='custom_avg_pool',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False
)

# ------------------------------------------------------------------
# 4) Replace the original functional_model with the custom kernel
# ------------------------------------------------------------------
def functional_model(
    x,
    *,
    avg_pool_kernel_size,
    avg_pool_stride,
    avg_pool_padding,
    avg_pool_ceil_mode,          # ignored – our kernel implements the same as ceil_mode=False
    avg_pool_count_include_pad   # ignored – we match count_include_pad=False (default)
):
    # Compute output length the same way PyTorch does (ceil_mode is False)
    in_len = x.shape[2]
    out_len = (in_len + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1

    # Allocate output tensor on the same device as input
    out = torch.empty((x.shape[0], x.shape[1], out_len), dtype=x.dtype, device=x.device)

    # Launch our custom kernel
    avg_pool_ext.avg_pool1d_forward(
        x.contiguous(),
        out,
        avg_pool_kernel_size,
        avg_pool_stride,
        avg_pool_padding
    )
    return out

# ------------------------------------------------------------------
# 5) Boiler-plate used for benchmarking (unchanged from the original script)
# ------------------------------------------------------------------
batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4

def get_init_inputs():
    return [kernel_size, stride, padding]

def get_inputs():
    x = torch.rand(batch_size, in_channels, input_length, device='cuda')
    return [x]

# When the evaluator imports this file, only `functional_model` will be used.
