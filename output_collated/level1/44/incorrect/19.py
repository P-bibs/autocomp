# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_114736/code_7.py
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
# CUDA source – optimized tiled average-pooling kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// BLOCK_SIZE is fixed per design to ensure efficient shared memory usage
constexpr int BLOCK_SIZE = 1024;

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
    // The number of output positions computed by this block
    const int TILE = BLOCK_SIZE - (kernel_size - 1);
    
    // Each block handles one channel of one batch element
    int b_c = blockIdx.x;
    int b = b_c / channels;
    int c = b_c % channels;

    int tile_idx = blockIdx.y;
    int out_start = tile_idx * TILE;
    int tid = threadIdx.x;

    // Shared memory for the input segment required to cover the current tile
    extern __shared__ float s_data[];

    // Load one element per thread into shared memory.
    // Each tile needs (TILE + K - 1) inputs to cover all window overlaps.
    int load_start = out_start * stride - padding;
    int load_idx = load_start + tid;
    
    if (load_idx >= 0 && load_idx < in_len) {
        s_data[tid] = input[(b * channels + c) * in_len + load_idx];
    } else {
        s_data[tid] = 0.0f;
    }
    __syncthreads();

    // Each thread computes one output element if it falls within the current tile
    if (tid < TILE) {
        int out_i = out_start + tid;
        if (out_i < out_len) {
            float sum = 0.0f;
            // The pooling window starts at tid relative to the loaded shared memory block
            for (int k = 0; k < kernel_size; ++k) {
                sum += s_data[tid + k];
            }

            float result;
            if (count_include_pad) {
                result = sum / static_cast<float>(kernel_size);
            } else {
                // Denominator adjusted for valid input range
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
}

void avg_pool1d_cuda(
    at::Tensor input,
    at::Tensor output,
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

    dim3 grid(batch * channels, n_tiles);
    dim3 threads(BLOCK_SIZE);
    size_t shared_mem = BLOCK_SIZE * sizeof(float);

    avg_pool1d_kernel<<<grid, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        in_len,
        out_len,
        kernel_size,
        stride,
        padding,
        count_include_pad
    );
}
"""

cpp_source = r"""
void avg_pool1d_cuda(at::Tensor input, at::Tensor output, int kernel, int stride, int pad, bool count_pad);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d", &avg_pool1d_cuda, "Optimized AvgPool1D");
}
"""

module = load_inline(
    name='avg_pool1d_optimized',
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
    L_in = x.shape[2]
    if avg_pool_ceil_mode:
        L_out = (L_in + 2 * avg_pool_padding - avg_pool_kernel_size + avg_pool_stride - 1) // avg_pool_stride + 1
    else:
        L_out = (L_in + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1
    
    out = torch.empty((x.shape[0], x.shape[1], L_out), dtype=x.dtype, device=x.device)
    
    module.avg_pool1d(
        x, out, 
        avg_pool_kernel_size, 
        avg_pool_stride, 
        avg_pool_padding, 
        avg_pool_count_include_pad
    )
    return out
