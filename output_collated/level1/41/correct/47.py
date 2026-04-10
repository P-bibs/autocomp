# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_041646/code_30.py
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

# Global variables for caching
fused_ext = None
gpu_tensors = {}

# CUDA kernel using shared memory tiling to reduce global memory bandwidth
# Each block processes one (batch, channel) slice
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void maxpool1d_tiled_kernel(
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
    // Each block handles one channel of a batch
    int b_c_idx = blockIdx.x;
    const float* in_ptr = input + (size_t)b_c_idx * input_length;
    float* out_ptr = output + (size_t)b_c_idx * output_length;

    // Shared memory: enough to hold a segment of the input sequence
    // Tile size logic: blockDim.x outputs need (blockDim.x-1)*stride + kernel_size*dilation input elements
    extern __shared__ float sdata[];

    for (int out_offset = 0; out_offset < output_length; out_offset += blockDim.x) {
        int out_idx = out_offset + threadIdx.x;
        
        // Load data into shared memory
        // We need inputs corresponding to this tile's range
        int thread_data_idx = out_offset * stride - padding + threadIdx.x;
        
        // Load input values into shared memory cooperatively
        // The number of elements to load per tile
        int load_idx = threadIdx.x;
        int max_load_idx = blockDim.x * stride + (kernel_size - 1) * dilation;
        
        while (load_idx < max_load_idx) {
            int input_pos = out_offset * stride - padding + load_idx;
            if (input_pos >= 0 && input_pos < input_length)
                sdata[load_idx] = in_ptr[input_pos];
            else
                sdata[load_idx] = -1e38f; // Representing -INFINITY
            load_idx += blockDim.x;
        }
        __syncthreads();

        if (out_idx < output_length) {
            float max_val = -1e38f;
            int input_start = out_idx * stride - padding;
            
            for (int k = 0; k < kernel_size; k++) {
                int s_idx = (input_start + k * dilation) - (out_offset * stride - padding);
                if (s_idx >= 0 && s_idx < max_load_idx) {
                    float val = sdata[s_idx];
                    if (val > max_val) max_val = val;
                }
            }
            out_ptr[out_idx] = max_val;
        }
        __syncthreads();
    }
}

void maxpool1d_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int output_length = output.size(2);
    
    int threads = 256;
    int blocks = batch_size * channels;
    
    // Shared memory size calculation
    int sdata_size = (threads * stride + (kernel_size - 1) * dilation) * sizeof(float);
    
    maxpool1d_tiled_kernel<<<blocks, threads, sdata_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
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
#include <torch/extension.h>
void maxpool1d_forward(const torch::Tensor& input, torch::Tensor& output, int kernel_size, int stride, int padding, int dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool1d", &maxpool1d_forward, "Tiled MaxPool1D");
}
"""

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    global fused_ext
    if fused_ext is None:
        fused_ext = load_inline(name='maxpool1d_tiled', cpp_sources=cpp_source, cuda_sources=cuda_kernel, extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)
    
    # Ensure input is contiguous and on GPU
    x_gpu = x.cuda().contiguous() if x.device.type != 'cuda' else x.contiguous()
    
    # Output calculation
    in_len = x_gpu.size(2)
    numerator = in_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1
    if maxpool_ceil_mode:
        output_length = (numerator + maxpool_stride - 1) // maxpool_stride + 1
    else:
        output_length = numerator // maxpool_stride + 1
        
    output = torch.empty((x_gpu.size(0), x_gpu.size(1), output_length), device=x_gpu.device, dtype=x_gpu.dtype)
    
    fused_ext.maxpool1d(x_gpu, output, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation)
    return output
