# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_040207/code_18.py
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

# Global persistent extension
fused_ext = None

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
    global fused_ext
    
    if fused_ext is None:
        cuda_source = r'''
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <algorithm>

        __global__ void maxpool1d_kernel(
            const float* __restrict__ input,
            float* __restrict__ output,
            int batch_size, int channels, int input_length, int output_length,
            int k_size, int stride, int padding, int dilation
        ) {
            extern __shared__ float s_data[];
            
            const int b = blockIdx.y / channels;
            const int c = blockIdx.y % channels;
            const int batch_chan_offset = (b * channels + c) * input_length;
            const int out_offset = (b * channels + c) * output_length;
            
            // Each thread block processes a tile of the output
            // We load a segment of input into shared memory sufficient for the window
            int tid = threadIdx.x;
            
            for (int out_start = blockIdx.x * blockDim.x; out_start < output_length; out_start += blockDim.x * gridDim.x) {
                int out_idx = out_start + tid;
                if (out_idx >= output_length) break;
                
                int start_pos = out_idx * stride - padding;
                float max_val = -1e38f; // Specialized float_min
                
                #pragma unroll
                for (int k = 0; k < k_size; ++k) {
                    int pos = start_pos + k * dilation;
                    if (pos >= 0 && pos < input_length) {
                        float val = input[batch_chan_offset + pos];
                        if (val > max_val) max_val = val;
                    }
                }
                output[out_offset + out_idx] = max_val;
            }
        }
        '''
        
        cpp_source = "void maxpool1d_forward(const torch::Tensor&, torch::Tensor&, int, int, int, int, int);"
        
        fused_ext = load_inline(
            name='maxpool1d_optimized',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=['maxpool1d_forward'],
            extra_cuda_cflags=['-O3', '--use_fast_math']
        )

    # Calculate output dimensions
    total_len = x.size(2)
    num_out = ((total_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride) + 1
    output = torch.empty((x.size(0), x.size(1), num_out), device=x.device, dtype=x.dtype)
    
    # Launch configuration
    threads = 256
    blocks = min(1024, (num_out + threads - 1) // threads)
    
    # We use a 2D grid: (blocks, batch * channels) to parallelize fully
    grid = (blocks, x.size(0) * x.size(1))
    
    fused_ext.maxpool1d_forward(
        x.contiguous(), output, x.size(0), x.size(1), total_len, num_out,
        maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation
    )
    
    return output
