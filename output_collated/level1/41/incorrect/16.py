# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_041646/code_29.py
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
# CUDA kernel – optimized tiled max‑pool 1D with shared memory
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void maxpool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int features,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Shared memory: each block needs space for the tile
    // tile_len = blockDim.x * stride + (kernel_size - 1) * dilation
    // We use a safe upper bound or dynamic shared memory. 
    // For 256 threads, stride 1, kernel 8, dil 3, max needed is ~280 floats.
    extern __shared__ float s_data[];

    const int tid = threadIdx.x;
    const int block_output_start = blockIdx.x * stride * blockDim.x;
    
    const int feature_idx = blockIdx.y;
    const int input_offset = feature_idx * input_length;
    const int output_offset = feature_idx * output_length;

    // Load tile into shared memory
    // Each thread loads multiple elements if necessary to cover the tile
    const int tile_len = (blockDim.x - 1) * stride + (kernel_size - 1) * dilation + 1;
    for (int i = tid; i < tile_len; i += blockDim.x) {
        int global_pos = block_output_start - padding + i;
        if (global_pos >= 0 && global_pos < input_length) {
            s_data[i] = __ldg(&input[input_offset + global_pos]);
        } else {
            s_data[i] = -FLT_MAX;
        }
    }
    __syncthreads();

    // Compute output
    const int out_idx = block_output_start + tid * stride;
    if (out_idx < output_length) {
        float max_val = -FLT_MAX;
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            int local_pos = tid * stride + k * dilation;
            float val = s_data[local_pos];
            if (val > max_val) max_val = val;
        }
        output[output_offset + out_idx] = max_val;
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
    int input_length = input.size(2);
    int output_length = output.size(2);
    int total_features = batch_size * features;

    const int threads = 256;
    const int blocks = (output_length + threads * stride - 1) / (threads * stride);
    
    // tile_len = (threads-1)*stride + (kernel-1)*dilation + 1
    const int tile_len = (threads - 1) * stride + (kernel_size - 1) * dilation + 1;

    maxpool1d_kernel<<<dim3(blocks, total_features), threads, tile_len * sizeof(float)>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        features,
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
void maxpool1d_cuda(torch::Tensor input, torch::Tensor output, int kernel_size, int stride, int padding, int dilation);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool1d", &maxpool1d_cuda, "Optimized MaxPool1d CUDA");
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
    
    # Reshape to treat (batch, feat) as one dim for the grid
    x_view = x.view(-1, seq_len)
    out_view = output.view(-1, out_len)
    
    maxpool_ext.maxpool1d(
        x_view, out_view, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation
    )
    
    return output
