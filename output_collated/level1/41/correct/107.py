# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_050814/code_12.py
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
# Global compiled extension (lazy initialisation)
# ----------------------------------------------------------------------
fused_ext = None

# ----------------------------------------------------------------------
# Optimised MaxPool1D: shared‑memory tiled kernel
# ----------------------------------------------------------------------
def functional_model(
    x,
    *,
    maxpool_kernel_size,
    maxpool_stride,
    maxpool_padding,
    maxpool_dilation,
    maxpool_ceil_mode,
    maxpool_return_indices,   # ignored – we only implement forward pass
):
    global fused_ext

    # ------------------------------------------------------------------
    # Compile the inline CUDA extension the first time functional_model
    # is called.
    # ------------------------------------------------------------------
    if fused_ext is None:
        cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <float.h>

constexpr int TILE_SIZE = 256;               // threads per block
constexpr int MAX_SHARED = 4096;             // maximal shared‑memory floats (≈16 KB)

__global__ void maxpool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // --------------------------------------------------------------
    // Shared memory tile – holds the input region needed by the block
    // --------------------------------------------------------------
    __shared__ float sdata[MAX_SHARED];

    // Output index where this block starts
    const int out_start = blockIdx.y * TILE_SIZE;

    // Determine batch & channel for this block
    int idx = blockIdx.x;
    int batch_idx = idx / channels;
    int channel_idx = idx % channels;

    // Base pointers for the current (batch, channel) slice
    const float* in_ptr = input +
        (static_cast<long long>(batch_idx) * channels + channel_idx) * input_length;
    float* out_ptr = output +
        (static_cast<long long>(batch_idx) * channels + channel_idx) * output_length;

    // --------------------------------------------------------------
    // Compute the range of input indices that must be loaded for the tile
    // --------------------------------------------------------------
    int start_input = out_start * stride - padding;
    if (start_input < 0) start_input = 0;

    int out_end = (out_start + TILE_SIZE < output_length) ?
                  (out_start + TILE_SIZE) : output_length;
    int last_out_pos = out_end - 1;

    int end_input = last_out_pos * stride - padding +
                    (kernel_size - 1) * dilation + 1;
    if (end_input > input_length) end_input = input_length;

    int load_count = end_input - start_input;   // number of floats to cache

    // --------------------------------------------------------------
    // Cooperative load of the input tile (coalesced global reads)
    // --------------------------------------------------------------
    for (int i = threadIdx.x; i < load_count; i += blockDim.x) {
        int in_idx = start_input + i;
        sdata[i] = __ldg(&in_ptr[in_idx]);     // read‑only cache hint
    }
    __syncthreads();

    // --------------------------------------------------------------
    // Compute max‑pooling for the outputs handled by this block
    // --------------------------------------------------------------
    for (int out_pos = out_start + threadIdx.x;
         out_pos < out_end;
         out_pos += blockDim.x) {

        int start_pos = out_pos * stride - padding;
        int offset_start = start_pos - start_input;   // offset inside the tile

        float max_val = -FLT_MAX;
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            int in_pos = offset_start + k * dilation;
            if (in_pos >= 0 && in_pos < load_count) {
                max_val = fmaxf(max_val, sdata[in_pos]);
            }
        }
        out_ptr[out_pos] = max_val;
    }
}

// ------------------------------------------------------------------
// Host‑side wrapper called from Python
// ------------------------------------------------------------------
void maxpool1d_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const int batch_size   = input.size(0);
    const int channels     = input.size(1);
    const int input_length = input.size(2);
    const int output_length = output.size(2);

    const int tile_size = TILE_SIZE;
    dim3 grid(batch_size * channels,
              (output_length + tile_size - 1) / tile_size);
    dim3 block(tile_size);

    maxpool1d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation
    );
}
"""
        cpp_source = r"""
#include <torch/extension.h>

void maxpool1d_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool1d", &maxpool1d_forward, "MaxPool1D forward");
}
"""
        fused_ext = load_inline(
            name="maxpool1d_lib",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_75"],
            with_cuda=True,
        )

    # ------------------------------------------------------------------
    # Compute output length (same formula as the original PyTorch code)
    # ------------------------------------------------------------------
    total_len = x.size(2)
    k = maxpool_kernel_size
    s = maxpool_stride
    p = maxpool_padding
    d = maxpool_dilation

    if maxpool_ceil_mode:
        output_length = ((total_len + 2 * p - d * (k - 1) - 1) + s - 1) // s + 1
    else:
        output_length = (total_len + 2 * p - d * (k - 1) - 1) // s + 1

    # ------------------------------------------------------------------
    # Allocate GPU tensors and run the custom kernel
    # ------------------------------------------------------------------
    x_gpu = x.contiguous().cuda()
    output = torch.empty(
        (x.size(0), x.size(1), output_length),
        device="cuda",
        dtype=x.dtype,
    )

    fused_ext.maxpool1d(x_gpu, output, k, s, p, d)

    return output
