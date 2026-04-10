# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_113602/code_3.py
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

# -------------------------------------------------------------------------
# CUDA source – tiled average‑pooling kernel + host launcher
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch,
    const int channels,
    const int in_len,
    const int out_len,
    const int kernel_size,
    const int stride,
    const int padding,
    const bool count_include_pad)
{
    // shared memory tile: size = blockDim.x + kernel_size - 1
    extern __shared__ float tile[];

    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int K = kernel_size;

    // ---------- block / channel / batch identification ----------
    const int blocksPerChan = (out_len + blockSize - 1) / blockSize;
    const int blockId = blockIdx.x;
    const int blockOffset = blockId % blocksPerChan;          // segment inside a channel
    const int channelBlock = blockId / blocksPerChan;         // which (batch,channel) pair
    const int b = channelBlock / channels;                    // batch index
    const int c = channelBlock % channels;                    // channel index

    // Guard against stray blocks (should never happen with correct launch config)
    if (b >= batch) return;

    // ---------- output position handled by this thread ----------
    const int out_start = blockOffset * blockSize;
    const int out_i = out_start + tid;   // may be >= out_len

    // ---------- input region that must be present in the tile ----------
    // global input index that corresponds to the first output of this block
    const int in_start = out_start * stride - padding;

    // ----- load primary tile elements -----
    int idx0 = in_start + tid;
    if (idx0 >= 0 && idx0 < in_len) {
        tile[tid] = input[(b * channels + c) * in_len + idx0];
    } else {
        tile[tid] = 0.0f;   // zero‑padding
    }

    // ----- load right‑halo (up to K‑1 extra values) -----
    if (tid < K - 1) {
        int idx1 = in_start + blockSize + tid;
        if (idx1 >= 0 && idx1 < in_len) {
            tile[blockSize + tid] = input[(b * channels + c) * in_len + idx1];
        } else {
            tile[blockSize + tid] = 0.0f;
        }
    }

    // Wait for the whole tile to be populated
    __syncthreads();

    // ---------- compute average (only if this thread corresponds to a real output) ----------
    if (out_i < out_len) {
        float sum = 0.0f;
        int valid = 0;

        for (int k = 0; k < K; ++k) {
            int globalIdx = in_start + tid + k;          // original global index
            if (globalIdx >= 0 && globalIdx < in_len) {
                sum   += tile[tid + k];
                ++valid;
            }
            // else: tile[tid+k] is already zero, nothing to add
        }

        float avg;
        if (count_include_pad) {
            // divisor is always kernel_size (padded zeros count)
            avg = sum / static_cast<float>(K);
        } else {
            // divisor is the number of valid positions
            avg = (valid > 0) ? (sum / static_cast<float>(valid)) : 0.0f;
        }

        output[(b * channels + c) * out_len + out_i] = avg;
    }
}

// Host‑side wrapper – called from Python
void fused_op_forward(
    int blocks,
    int threads,
    const torch::Tensor &input,
    torch::Tensor &output,
    int batch,
    int channels,
    int in_len,
    int out_len,
    int kernel_size,
    int stride,
    int padding,
    bool ceil_mode,          // not used inside the kernel (output size already computed)
    bool count_include_pad,
    size_t shared_mem)
{
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    avg_pool1d_kernel<<<blocks, threads, shared_mem>>>(
        in_ptr, out_ptr,
        batch, channels, in_len, out_len,
        kernel_size, stride, padding,
        count_include_pad);
}
"""

# -------------------------------------------------------------------------
# C++ binding – exposes the wrapper to Python
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    int blocks,
    int threads,
    const torch::Tensor &input,
    torch::Tensor &output,
    int batch,
    int channels,
    int in_len,
    int out_len,
    int kernel_size,
    int stride,
    int padding,
    bool ceil_mode,
    bool count_include_pad,
    size_t shared_mem);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused average‑pooling forward");
}
"""

# -------------------------------------------------------------------------
# Compile the extension (Python will import the module `fused_ext`)
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True)

# -------------------------------------------------------------------------
# Original helper functions (kept unchanged)
# -------------------------------------------------------------------------
batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4

def get_init_inputs():
    return [kernel_size, stride, padding]

def get_inputs():
    x = torch.rand(batch_size, in_channels, input_length)
    return [x]

# -------------------------------------------------------------------------
# Optimised functional_model – uses the tiled CUDA kernel
# -------------------------------------------------------------------------
def functional_model(
    x,
    *,
    avg_pool_kernel_size,
    avg_pool_stride,
    avg_pool_padding,
    avg_pool_ceil_mode,
    avg_pool_count_include_pad):
    """
    Performs 1‑D average pooling with the given parameters.
    The implementation caches a tile of the input in shared memory,
    reducing redundant global‑memory traffic.
    """
    # Ensure the input lives on the GPU
    if not x.is_cuda:
        x = x.cuda()

    batch, channels, in_len = x.shape

    # ----- compute output length (same formula as PyTorch) -----
    if avg_pool_ceil_mode:
        out_len = (in_len + 2 * avg_pool_padding - avg_pool_kernel_size + avg_pool_stride - 1) // avg_pool_stride + 1
    else:
        out_len = (in_len + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1

    # Allocate output tensor
    output = torch.empty(batch, channels, out_len, dtype=x.dtype, device=x.device)

    # ----- launch config -----
    BLOCK = 1024                               # threads per block
    blocks_per_channel = (out_len + BLOCK - 1) // BLOCK
    total_blocks = batch * channels * blocks_per_channel

    # shared memory = (BLOCK + kernel_size - 1) floats
    shared_mem = (BLOCK + avg_pool_kernel_size - 1) * 4   # 4 bytes per float

    # ----- call the compiled CUDA kernel -----
    fused_ext.fused_op(
        total_blocks,
        BLOCK,
        x,
        output,
        batch,
        channels,
        in_len,
        out_len,
        avg_pool_kernel_size,
        avg_pool_stride,
        avg_pool_padding,
        avg_pool_ceil_mode,
        avg_pool_count_include_pad,
        shared_mem)

    return output
