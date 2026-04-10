# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_114736/code_3.py
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

# ----------------------------------------------------------------------
# CUDA source – the tiled average-pooling kernel that uses shared memory
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 1024;          // threads per block

__global__ void avg_pool1d_kernel(
    const float* __restrict__ input,      // (B, C, L_in)
    float* __restrict__ output,           // (B, C, L_out)
    const int batch,
    const int channels,
    const int in_len,
    const int out_len,
    const int kernel_size,
    const int stride,
    const int padding,
    const bool count_include_pad)
{
    // number of output positions computed by this block
    const int TILE = BLOCK_SIZE - (kernel_size - 1);   // =1017 for K=8

    // ---- which (batch, channel) pair this block is responsible for ----
    int ch_block = blockIdx.x;                         // 0 … (B*C‑1)
    int b = ch_block / channels;
    int c = ch_block % channels;

    // ---- which tile of the length dimension -----------------------
    int tile_idx = blockIdx.y;
    int out_start = tile_idx * TILE;                   // first output index in this block

    int tid = threadIdx.x;

    // ---- shared memory for the input segment --------------------
    extern __shared__ float s_data[];                  // size = BLOCK_SIZE

    // ---- load one element per thread (including the K‑1 overlap) ----
    int load_idx = out_start + tid - padding;
    if (load_idx >= 0 && load_idx < in_len) {
        s_data[tid] = input[(b * channels + c) * in_len + load_idx];
    } else {
        // padded zero – we keep it as zero (counted or not depends on flag)
        s_data[tid] = 0.0f;
    }
    __syncthreads();

    // ---- each thread computes one output (if inside the tile) ----
    if (tid < TILE) {
        int out_i = out_start + tid;
        if (out_i >= out_len) return;

        // sum the K values from shared memory
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; ++k) {
            sum += s_data[tid + k];
        }

        float result;
        if (count_include_pad) {
            // denominator is always the full kernel size
            result = sum / static_cast<float>(kernel_size);
        } else {
            // count how many positions are actually inside the tensor
            int input_start = out_i * stride - padding;
            int denom = 0;
            for (int k = 0; k < kernel_size; ++k) {
                int idx = input_start + k;
                if (idx >= 0 && idx < in_len) ++denom;
            }
            result = (denom > 0) ? sum / static_cast<float>(denom) : 0.0f;
        }

        output[(b * channels + c) * out_len + out_i] = result;
    }
}

// ------------------------------------------------------------------
// Host-side wrapper that computes grid dimensions and launches the kernel
// ------------------------------------------------------------------
void avg_pool1d_cuda(
    at::Tensor input,      // (B, C, L_in)
    at::Tensor output,    // (B, C, L_out)
    int kernel_size,
    int stride,
    int padding,
    bool count_include_pad)
{
    const int batch    = input.size(0);
    const int channels = input.size(1);
    const int in_len   = input.size(2);
    const int out_len  = output.size(2);

    const int TILE    = BLOCK_SIZE - (kernel_size - 1);
    const int n_tiles = (out_len + TILE - 1) / TILE;

    const dim3 blocks(batch * channels, n_tiles);   // 2‑D grid
    const dim3 threads(BLOCK_SIZE);                // 1‑D block

    const size_t shared_mem = BLOCK_SIZE * sizeof(float);

    avg_pool1d_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        in_len,
        out_len,
        kernel_size,
        stride,
        padding,
        count_include_pad);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}
"""

# ----------------------------------------------------------------------
# C++ binding – exposes the CUDA function to Python
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void avg_pool1d_cuda(
    at::Tensor input,
    at::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    bool count_include_pad);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d", &avg_pool1d_cuda,
          "Tiled average pooling 1D (CUDA)");
}
"""

# ----------------------------------------------------------------------
# Compile the extension (done once; the result is cached)
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='avg_pool1d_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional model – replaces the original F.avg_pool1d call
# ----------------------------------------------------------------------
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
    Average-pooling 1D implemented with a tiled CUDA kernel that
    caches a block of input in shared memory, dramatically reducing
    global-memory traffic.
    """
    # ----- compute output length (same formula as PyTorch) -----
    L_in = x.shape[2]
    if avg_pool_ceil_mode:
        # ceil((L_in + 2*pad - kernel) / stride) + 1
        L_out = (L_in + 2 * avg_pool_padding - avg_pool_kernel_size + avg_pool_stride - 1) // avg_pool_stride + 1
    else:
        # floor((L_in + 2*pad - kernel) / stride) + 1
        L_out = (L_in + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1

    # ----- allocate output tensor -----
    out = torch.empty((x.shape[0], x.shape[1], L_out),
                      dtype=x.dtype, device=x.device)

    # ----- launch our tiled CUDA kernel -----
    fused_ext.avg_pool1d(
        x, out,
        avg_pool_kernel_size,
        avg_pool_stride,
        avg_pool_padding,
        avg_pool_count_include_pad)

    return out
