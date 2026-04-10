# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_050814/code_8.py
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

# Global variable to store compiled extension
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
        cuda_source = r"""
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <algorithm>
        #include <float.h>

        // Optimized kernel with shared memory tiling and collaborative loading
        __global__ void maxpool1d_kernel_optimized(
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
            // Shared memory for input window caching
            extern __shared__ float smem[];
            
            int batch_idx = blockIdx.z;
            int channel_idx = blockIdx.y;
            int block_output_start = blockIdx.x * blockDim.x;
            int tid = threadIdx.x;
            int global_output_idx = block_output_start + tid;
            
            // Bounds check for output
            if (global_output_idx >= output_length) return;
            
            // Input/output pointers for this batch/channel
            const float* in_ptr = input + (batch_idx * channels + channel_idx) * input_length;
            float* out_ptr = output + (batch_idx * channels + channel_idx) * output_length;
            
            // Calculate input window bounds for this output position
            int start_pos = global_output_idx * stride - padding;
            int end_pos = start_pos + (kernel_size - 1) * dilation;
            
            // Determine shared memory window bounds
            int window_start = max(0, start_pos);
            int window_end = min(input_length - 1, end_pos);
            int window_size = window_end - window_start + 1;
            
            // Collaborative loading of input window into shared memory
            for (int i = tid; i < window_size; i += blockDim.x) {
                int input_idx = window_start + i;
                if (input_idx >= 0 && input_idx < input_length) {
                    smem[i] = in_ptr[input_idx];
                } else {
                    smem[i] = -FLT_MAX;
                }
            }
            __syncthreads();
            
            // Compute maxpool for this output position using shared memory
            float max_val = -FLT_MAX;
            for (int k = 0; k < kernel_size; ++k) {
                int input_pos = start_pos + k * dilation;
                if (input_pos >= 0 && input_pos < input_length) {
                    int smem_idx = input_pos - window_start;
                    max_val = fmaxf(max_val, smem[smem_idx]);
                }
            }
            
            out_ptr[global_output_idx] = max_val;
        }

        void maxpool1d_forward(
            const torch::Tensor& input,
            torch::Tensor& output,
            int kernel_size,
            int stride,
            int padding,
            int dilation
        ) {
            const int batch_size = input.size(0);
            const int channels = input.size(1);
            const int input_length = input.size(2);
            const int output_length = output.size(2);
            
            // Compute optimal block and grid dimensions
            const int threads_per_block = 256;
            const int blocks_per_channel = (output_length + threads_per_block - 1) / threads_per_block;
            
            dim3 grid(blocks_per_channel, channels, batch_size);
            dim3 block(threads_per_block);
            
            // Calculate shared memory size needed
            int max_window_elements = kernel_size * dilation + 2 * padding;
            size_t shared_mem_size = max_window_elements * sizeof(float);
            
            maxpool1d_kernel_optimized<<<grid, block, shared_mem_size>>>(
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
            m.def("maxpool1d", &maxpool1d_forward, "Optimized MaxPool1D forward with shared memory");
        }
        """

        fused_ext = load_inline(
            name='maxpool1d_lib_optimized',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
            with_cuda=True
        )

    # Calculate output dimensions
    total_len = x.size(2)
    k, s, p, d = maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation
    
    if maxpool_ceil_mode:
        output_length = ((total_len + 2 * p - d * (k - 1) - 1) + s - 1) // s + 1
    else:
        output_length = (total_len + 2 * p - d * (k - 1) - 1) // s + 1
    
    # Ensure contiguous GPU input
    x_gpu = x.contiguous().cuda()
    output = torch.empty((x.size(0), x.size(1), output_length), device='cuda', dtype=x.dtype)
    
    fused_ext.maxpool1d(x_gpu, output, k, s, p, d)
    
    return output
