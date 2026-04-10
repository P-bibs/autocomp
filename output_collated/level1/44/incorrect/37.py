# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_120345/code_3.py
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

# --- Optimized CUDA Kernel ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h> // For checking CUDA errors

// Helper macro for CUDA error checking
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Fused kernel: cumsum + gather + compute averages
__global__ void fused_avg_pool1d_kernel(
    const float* __restrict__ input, // [B, C, L_in]
    float* __restrict__ output,      // [B, C, L_out]
    const int64_t B,
    const int64_t C,
    const int64_t L_in,
    const int64_t L_out,
    const int64_t kernel_size,
    const int64_t stride,
    const int64_t padding,
    const bool ceil_mode,
    const bool count_include_pad
) {
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x; // Global output index
    if (i >= B * C * L_out) return;

    const int64_t b = i / (C * L_out);
    const int64_t c = (i / L_out) % C;
    const int64_t out_pos = i % L_out;

    const int64_t start_idx = out_pos * stride - padding;
    const int64_t end_idx_excl = start_idx + kernel_size;

    const float* x = input + b * C * L_in + c * L_in;

    float sum = 0.0f;
    int64_t valid_count = 0;

    for (int64_t idx = start_idx; idx < end_idx_excl; idx++) {
        if (idx >= 0 && idx < L_in) {
            sum += x[idx];
            valid_count++;
        }
    }

    float avg = 0.0f;
    if (count_include_pad) {
        if (kernel_size > 0) {
            avg = sum / static_cast<float>(kernel_size);
        }
    } else {
        if (valid_count > 0) {
            avg = sum / static_cast<float>(valid_count);
        }
    }

    output[i] = avg;
}

void launch_fused_avg_pool1d(
    const torch::Tensor& input,
    torch::Tensor& output,
    const int64_t kernel_size,
    const int64_t stride,
    const int64_t padding,
    const bool ceil_mode,
    const bool count_include_pad
) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);

    const auto sizes = input.sizes();
    const int64_t B = sizes[0];
    const int64_t C = sizes[1];
    const int64_t L_in = sizes[2];
    const int64_t L_out = output.size(2);

    const int64_t total_threads = B * C * L_out;
    const int64_t threads_per_block = 256;
    const int64_t blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    fused_avg_pool1d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, L_in, L_out,
        kernel_size, stride, padding, ceil_mode, count_include_pad
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void launch_fused_avg_pool1d(
    const torch::Tensor& input,
    torch::Tensor& output,
    const int64_t kernel_size,
    const int64_t stride,
    const int64_t padding,
    const bool ceil_mode,
    const bool count_include_pad
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_avg_pool1d", &launch_fused_avg_pool1d, "Fused Avg Pool 1D forward pass");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_avg_pool1d_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
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
    """
    Functional model using a custom CUDA kernel for optimized 1D average pooling.
    This avoids built-in PyTorch convolution/matmul functions and uses a single
    fused CUDA kernel for memory-efficient computation.
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert x.dtype == torch.float32, "Only float32 supported in this implementation"

    # --- Output length calculation (PyTorch-style) ---------------------
    L_in = x.size(-1)
    if avg_pool_ceil_mode:
        L_out = (L_in + 2 * avg_pool_padding - avg_pool_kernel_size + avg_pool_stride - 1) // avg_pool_stride + 1
    else:
        L_out = (L_in + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1

    output_shape = (x.size(0), x.size(1), L_out)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)

    # Launch the fused CUDA kernel
    fused_ext.fused_avg_pool1d(
        x, out,
        avg_pool_kernel_size,
        avg_pool_stride,
        avg_pool_padding,
        avg_pool_ceil_mode,
        avg_pool_count_include_pad
    )
    return out
