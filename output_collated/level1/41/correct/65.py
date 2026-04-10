# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_042936/code_26.py
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

# Optimization: Use Shared Memory for tiled MaxPool1D
# Block size: 128 threads: [128, 1]. Each block handles a tile of the output.
# Shared memory will store the required input window for the tile.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <float.h>

#define TILE_WIDTH 128
#define MAX_SHARED_MEM 48000 // Conservative per-block shared memory limit

__global__ void maxpool1d_kernel_shared(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int features,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    extern __shared__ float s_input[];

    int feat_idx = blockIdx.y;
    int batch_idx = feat_idx / features;
    int f_idx = feat_idx % features;
    int tile_idx = blockIdx.x;
    
    int tile_start = tile_idx * TILE_WIDTH;
    
    // Calculate range of input needed for this output tile
    int input_window_start = tile_start * stride - padding;
    int input_window_end = (tile_start + TILE_WIDTH - 1) * stride - padding + (kernel_size - 1) * dilation;
    
    // Clamping the needed input indices
    int load_start = max(0, input_window_start);
    int load_end = min(input_length - 1, input_window_end);
    
    int input_base = (batch_idx * features + f_idx) * input_length;

    // Collaborative loading into shared memory
    for (int i = threadIdx.x; i <= (load_end - load_start); i += blockDim.x) {
        s_input[i] = input[input_base + load_start + i];
    }
    __syncthreads();

    // Maxpool processing using shared memory
    int tid = threadIdx.x;
    if (tile_start + tid < output_length) {
        int out_idx = tile_start + tid;
        int start_pos = out_idx * stride - padding;
        
        float max_val = -FLT_MAX;
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            int pos = start_pos + k * dilation;
            if (pos >= load_start && pos <= load_end) {
                float val = s_input[pos - load_start];
                if (val > max_val) max_val = val;
            }
        }
        output[(batch_idx * features + f_idx) * output_length + out_idx] = max_val;
    }
}

void maxpool1d_cuda(
    torch::Tensor input,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int batch_size = input.size(0);
    int features = input.size(1);
    int output_length = output.size(2);
    
    dim3 threads(128);
    dim3 blocks((output_length + TILE_WIDTH - 1) / TILE_WIDTH, batch_size * features);
    
    // Shared memory size calculation: 
    // Worst case width is TILE_WIDTH * stride + (kernel_size-1)*dilation
    int shared_mem_size = (TILE_WIDTH * stride + (kernel_size) * dilation) * sizeof(float);
    
    maxpool1d_kernel_shared<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        features,
        input.size(2),
        output_length,
        kernel_size,
        stride,
        padding,
        dilation
    );
}
"""

cpp_source = r"""
void maxpool1d_cuda(torch::Tensor input, torch::Tensor output, int kernel_size, int stride, int padding, int dilation);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool1d", &maxpool1d_cuda, "Optimized Shared-Memory MaxPool1d CUDA");
}
"""

maxpool_ext = load_inline(
    name='maxpool_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

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
    if maxpool_return_indices:
        raise NotImplementedError("Indices not supported")
    
    x = x.contiguous()
    batch, feat, seq_len = x.shape
    
    if maxpool_ceil_mode:
        out_len = (seq_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1 + maxpool_stride - 1) // maxpool_stride + 1
    else:
        out_len = (seq_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
        
    output = torch.empty((batch, feat, out_len), device=x.device, dtype=x.dtype)
    
    maxpool_ext.maxpool1d(
        x, output, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation
    )
    
    return output
