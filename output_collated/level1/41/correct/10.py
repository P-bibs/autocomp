# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_035026/code_10.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['kernel_size', 'stride', 'padding', 'dilation', 'return_indices']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['maxpool_kernel_size', 'maxpool_stride', 'maxpool_padding', 'maxpool_dilation', 'maxpool_ceil_mode', 'maxpool_return_indices']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 1D.
    """

    def __init__(self, kernel_size: int, stride: int=None, padding: int=0, dilation: int=1, return_indices: bool=False):
        """
        Initializes the Max Pooling 1D layer.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
            padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices)

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
    # State for maxpool (nn.MaxPool1d)
    state_kwargs['maxpool_kernel_size'] = model.maxpool.kernel_size
    state_kwargs['maxpool_stride'] = model.maxpool.stride
    state_kwargs['maxpool_padding'] = model.maxpool.padding
    state_kwargs['maxpool_dilation'] = model.maxpool.dilation
    state_kwargs['maxpool_ceil_mode'] = model.maxpool.ceil_mode
    state_kwargs['maxpool_return_indices'] = model.maxpool.return_indices
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
# Optimised CUDA kernel – uses shared-memory tile and coalesced loads
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_pool1d_dilated_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    // Size of the shared-memory tile for one output chunk
    const int tile_len = blockDim.x * stride + (kernel_size - 1) * dilation;
    extern __shared__ float sdata[];

    // Which (batch, channel) does this block serve?
    const int bs = blockIdx.x / channels;
    const int ch = blockIdx.x % channels;
    if (bs >= batch_size) return;

    // Iterate over the output sequence in chunks of blockDim.x
    for (int out_start = 0; out_start < output_length; out_start += blockDim.x) {
        // Starting input index for this chunk
        const int in_start_block = out_start * stride - padding;

        // ------------------------------------------------------------------
        // Coalesced load of the required input tile into shared memory
        // ------------------------------------------------------------------
        for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
            const int input_pos = in_start_block + i;
            if (input_pos >= 0 && input_pos < input_length) {
                const int offset = ((bs * channels + ch) * input_length) + input_pos;
                sdata[i] = input[offset];
            } else {
                sdata[i] = -INFINITY;               // padding
            }
        }
        __syncthreads();

        // ------------------------------------------------------------------
        // Compute max pooling for the thread’s own output position (if any)
        // ------------------------------------------------------------------
        const int out_pos = out_start + threadIdx.x;
        if (out_pos < output_length) {
            float max_val = -INFINITY;
            for (int k = 0; k < kernel_size; ++k) {
                const int tile_idx = threadIdx.x * stride + k * dilation;
                float val = sdata[tile_idx];
                if (val > max_val) max_val = val;
            }
            const int out_idx = ((bs * channels + ch) * output_length) + out_pos;
            output[out_idx] = max_val;
        }
        // Ensure the next chunk loads fresh data
        __syncthreads();
    }
}

void max_pool1d_dilated_forward(
    const torch::Tensor input,
    torch::Tensor output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    const int batch_size = input.size(0);
    const int channels   = input.size(1);
    const int input_len  = input.size(2);
    const int out_len    = output.size(2);

    const int threads_per_block = 256;
    const int blocks = batch_size * channels;                // one block per (batch, channel)

    const int tile_len = threads_per_block * stride + (kernel_size - 1) * dilation;
    const size_t shared_mem = tile_len * sizeof(float);

    max_pool1d_dilated_kernel<<<blocks, threads_per_block, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_len,
        out_len,
        kernel_size,
        stride,
        padding,
        dilation
    );
}
"""

# ----------------------------------------------------------------------
# C++ binding (PYBIND11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void max_pool1d_dilated_forward(
    const torch::Tensor input,
    torch::Tensor output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pool1d_dilated_forward", &max_pool1d_dilated_forward,
          "Max-pooling 1D with dilation (forward)");
}
"""

# ----------------------------------------------------------------------
# Compile the extension
# ----------------------------------------------------------------------
maxpool_cuda = load_inline(
    name='max_pool1d_dilated',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional wrapper that will be imported / evaluated
# ----------------------------------------------------------------------
def functional_model(
    x,
    *,
    maxpool_kernel_size,
    maxpool_stride,
    maxpool_padding,
    maxpool_dilation,
    maxpool_ceil_mode,
    maxpool_return_indices,
):
    # ---- Compute output length (same formula as the original code) ----
    if maxpool_ceil_mode:
        output_length = int(
            torch.ceil(
                torch.tensor(
                    (x.shape[2] + 2 * maxpool_padding -
                     maxpool_dilation * (maxpool_kernel_size - 1) - 1) /
                    maxpool_stride + 1
                )
            ).item()
        )
    else:
        output_length = int(
            torch.floor(
                torch.tensor(
                    (x.shape[2] + 2 * maxpool_padding -
                     maxpool_dilation * (maxpool_kernel_size - 1) - 1) /
                    maxpool_stride + 1
                )
            ).item()
        )

    # ---- Allocate output tensor ----
    output = torch.empty(
        x.shape[0], x.shape[1], output_length,
        device=x.device, dtype=x.dtype
    )

    # ---- Launch the optimized CUDA kernel ----
    maxpool_cuda.max_pool1d_dilated_forward(
        x,
        output,
        maxpool_kernel_size,
        maxpool_stride,
        maxpool_padding,
        maxpool_dilation,
    )

    # ---- Return value(s) -------------------------------------------------
    if maxpool_return_indices:
        # The current kernel does not compute indices; return None for compatibility
        return output, None
    else:
        return output


# ----------------------------------------------------------------------
# Example parameters (the same ones used for timing the original version)
# ----------------------------------------------------------------------
batch_size = 64
features = 192
sequence_length = 65536
kernel_size = 8
stride = 1
padding = 4
dilation = 3
return_indices = False


def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]


def get_inputs():
    x = torch.rand(batch_size, features, sequence_length, device='cuda')
    return [x]
