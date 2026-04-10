# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_042936/code_8.py
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

# Global variables to store compiled kernel and device tensors
fused_ext = None
gpu_tensors = {}

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
    global fused_ext, gpu_tensors
    
    if fused_ext is None:
        cuda_source = r"""
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <algorithm>

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
            extern __shared__ float shared_input[];
            
            int batch_idx = blockIdx.z;
            int channel_idx = blockIdx.y;
            int block_output_start = blockIdx.x * blockDim.x;
            
            int tid = threadIdx.x;
            int threads_per_block = blockDim.x;
            
            const float* in_ptr = input + (batch_idx * channels + channel_idx) * input_length;
            float* out_ptr = output + (batch_idx * channels + channel_idx) * output_length;
            
            // Calculate the range of input indices this block needs
            int first_output = block_output_start;
            int last_output = min(first_output + threads_per_block - 1, output_length - 1);
            
            // Calculate input range needed for these outputs
            int input_start = first_output * stride - padding;
            int input_end = (last_output * stride - padding) + (kernel_size - 1) * dilation;
            
            // Load shared memory with required input data
            for (int i = tid; i <= input_end - input_start; i += threads_per_block) {
                int input_idx = input_start + i;
                if (input_idx >= 0 && input_idx < input_length) {
                    shared_input[i] = in_ptr[input_idx];
                } else {
                    shared_input[i] = -3.402823466e+38F; // -FLT_MAX
                }
            }
            __syncthreads();
            
            // Each thread computes one output
            int out_idx = block_output_start + tid;
            if (out_idx < output_length) {
                int start_pos = out_idx * stride - padding;
                float max_val = -3.402823466e+38F; // -FLT_MAX
                
                #pragma unroll 4
                for (int k = 0; k < kernel_size; ++k) {
                    int in_pos = start_pos + k * dilation;
                    if (in_pos >= 0 && in_pos < input_length) {
                        float val = shared_input[in_pos - input_start];
                        if (val > max_val) max_val = val;
                    }
                }
                out_ptr[out_idx] = max_val;
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
            int input_length = input.size(2);
            int output_length = output.size(2);
            
            // Use 256 threads per block
            int threads_per_block = 256;
            int blocks_per_channel = (output_length + threads_per_block - 1) / threads_per_block;
            
            // Shared memory size: enough to hold the input tile needed by each block
            int max_input_per_block = (threads_per_block - 1) * stride + (kernel_size - 1) * dilation + 1;
            size_t shared_mem_size = max_input_per_block * sizeof(float);
            
            dim3 grid(blocks_per_channel, channels, batch_size);
            dim3 block(threads_per_block);
            
            maxpool1d_kernel<<<grid, block, shared_mem_size>>>(
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
        void maxpool1d_forward(const torch::Tensor& input, torch::Tensor& output, int k, int s, int p, int d);
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("maxpool1d", &maxpool1d_forward, "MaxPool1D forward");
        }
        """

        fused_ext = load_inline(
            name='maxpool1d_lib',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            with_cuda=True
        )

    # Calculate output dimensions
    total_len = x.size(2)
    if maxpool_ceil_mode:
        output_length = ((total_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) + maxpool_stride - 1) // maxpool_stride + 1
    else:
        output_length = (total_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
    
    # Ensure contiguous GPU input
    x_gpu = x.cuda().contiguous()
    output = torch.empty((x.size(0), x.size(1), output_length), device='cuda', dtype=x.dtype)
    
    fused_ext.maxpool1d(x_gpu, output, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation)
    
    return output
