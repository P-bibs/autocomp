# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_044333/code_0.py
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
            int tid = threadIdx.x;
            
            if (batch_idx >= batch_size || channel_idx >= channels) return;
            
            // Calculate the range of output indices this block will process
            int block_output_start = blockIdx.x * blockDim.x;
            int block_output_end = min(block_output_start + blockDim.x, output_length);
            
            // Calculate the range of input indices needed for these outputs
            int first_output = block_output_start;
            int last_output = block_output_end - 1;
            
            int first_input_needed = first_output * stride - padding;
            int last_input_needed = last_output * stride - padding + (kernel_size - 1) * dilation;
            
            // Clamp to valid input range
            first_input_needed = max(0, first_input_needed);
            last_input_needed = min(input_length - 1, last_input_needed);
            
            int shared_size = last_input_needed - first_input_needed + 1;
            
            // Base pointer for this batch/channel
            const float* input_base = input + (batch_idx * channels + channel_idx) * input_length;
            float* output_base = output + (batch_idx * channels + channel_idx) * output_length;
            
            // Cooperatively load input data into shared memory
            for (int i = tid; i < shared_size; i += blockDim.x) {
                shared_input[i] = input_base[first_input_needed + i];
            }
            __syncthreads();
            
            // Process outputs assigned to this block
            for (int out_pos = block_output_start + tid; out_pos < block_output_end; out_pos += blockDim.x) {
                int start_pos = out_pos * stride - padding;
                float max_val = -FLT_MAX;
                
                #pragma unroll
                for (int k = 0; k < kernel_size; ++k) {
                    int in_pos = start_pos + k * dilation;
                    if (in_pos >= 0 && in_pos < input_length) {
                        // Access from shared memory
                        int shared_idx = in_pos - first_input_needed;
                        if (shared_idx >= 0 && shared_idx < shared_size) {
                            max_val = fmaxf(max_val, shared_input[shared_idx]);
                        }
                    }
                }
                output_base[out_pos] = max_val;
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
            const int batch_size = input.size(0);
            const int channels = input.size(1);
            const int input_length = input.size(2);
            const int output_length = output.size(2);
            
            // Use a reasonable block size
            const int BLOCK_SIZE = 256;
            dim3 block(BLOCK_SIZE);
            
            // Grid dimensions: (output_tiles, channels, batches)
            dim3 grid(
                (output_length + BLOCK_SIZE - 1) / BLOCK_SIZE,
                channels,
                batch_size
            );
            
            // Calculate shared memory size
            // Worst case: all inputs in the tile's receptive field
            int max_shared_elements = BLOCK_SIZE * stride + (kernel_size - 1) * dilation + 1;
            size_t shared_mem_size = max_shared_elements * sizeof(float);
            
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
