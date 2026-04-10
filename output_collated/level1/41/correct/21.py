# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_040207/code_6.py
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

fused_ext = None

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    global fused_ext
    if fused_ext is None:
        cuda_source = r"""
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <c10/cuda/CUDAGuard.h>

        __global__ void maxpool1d_shared_kernel(
            const float* __restrict__ input,
            float* __restrict__ output,
            const int batch_size,
            const int channels,
            const int input_length,
            const int output_length,
            const int k_size,
            const int stride,
            const int padding,
            const int dilation
        ) {
            // Use shared memory for caching input data
            extern __shared__ float s_data[];

            const int tid = threadIdx.x;
            const int bid = blockIdx.x;
            const int threads_per_block = blockDim.x;

            // Each block handles multiple output positions for better reuse
            const int items_per_block = threads_per_block;
            const int start_out_pos = bid * items_per_block;
            
            // Determine the range of input needed by this block
            const int first_out = start_out_pos;
            const int last_out = min(first_out + items_per_block - 1, output_length - 1);
            
            if (first_out >= output_length) return;

            // Calculate input range required for these outputs
            const int min_out_pos = first_out;
            const int max_out_pos = last_out;
            const int min_input_start = min_out_pos * stride - padding;
            const int max_input_end = (max_out_pos * stride - padding) + (k_size - 1) * dilation;

            const int shared_size = max_input_end - min_input_start + 1;
            const int base_batch_channel_offset = ((blockIdx.y * gridDim.z) + blockIdx.z) * input_length;

            // Load input data into shared memory cooperatively
            for (int i = tid; i < shared_size; i += threads_per_block) {
                const int input_idx = min_input_start + i;
                if (input_idx >= 0 && input_idx < input_length) {
                    s_data[i] = input[base_batch_channel_offset + input_idx];
                } else {
                    s_data[i] = -1e38f; // Padding value for max pooling
                }
            }
            __syncthreads();

            // Each thread computes one output element
            const int out_pos = start_out_pos + tid;
            if (out_pos < output_length) {
                const int input_start = out_pos * stride - padding;
                const int shared_start = input_start - min_input_start;

                float max_val = -1e38f;
                for (int k = 0; k < k_size; ++k) {
                    const int shared_idx = shared_start + k * dilation;
                    if (shared_idx >= 0 && shared_idx < shared_size) {
                        const float val = s_data[shared_idx];
                        max_val = fmaxf(max_val, val);
                    }
                }
                const int out_index = ((blockIdx.y * gridDim.z) + blockIdx.z) * output_length + out_pos;
                output[out_index] = max_val;
            }
        }

        void maxpool1d_forward(
            const torch::Tensor& input,
            torch::Tensor& output,
            int k_size,
            int stride,
            int padding,
            int dilation
        ) {
            // Ensure we're on the correct device
            const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
            
            const int batch_size = input.size(0);
            const int channels = input.size(1);
            const int input_length = input.size(2);
            const int output_length = output.size(2);

            // Launch configuration
            const int threads_per_block = 256;
            const int items_per_block = threads_per_block;
            const int num_blocks_x = (output_length + items_per_block - 1) / items_per_block;
            const int num_blocks_y = batch_size;
            const int num_blocks_z = channels;

            // Shared memory size calculation
            const int max_window_span = (k_size - 1) * dilation + 1;
            const int shared_memory_per_item = max_window_span + (items_per_block - 1) * stride;
            const int shared_memory_bytes = shared_memory_per_item * sizeof(float);

            dim3 grid(num_blocks_x, num_blocks_y, num_blocks_z);
            dim3 block(threads_per_block);

            maxpool1d_shared_kernel<<<grid, block, shared_memory_bytes>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                channels,
                input_length,
                output_length,
                k_size,
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
            int k_size,
            int stride,
            int padding,
            int dilation
        );

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("maxpool1d", &maxpool1d_forward, "Optimized MaxPool1D forward with shared memory");
        }
        """
        
        fused_ext = load_inline(
            name='fused_maxpool1d_opt',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            with_cuda=True
        )

    # Move tensor to GPU if necessary
    if not x.is_cuda:
        x = x.cuda()
    
    # Calculate output dimensions
    if maxpool_ceil_mode:
        output_length = int(torch.ceil(
            torch.tensor((x.size(2) + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) 
                        / float(maxpool_stride) + 1)
        ).item())
    else:
        output_length = (x.size(2) + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
    
    # Create output tensor
    output = torch.empty((x.size(0), x.size(1), output_length), device=x.device, dtype=x.dtype)
    
    # Launch optimized kernel
    fused_ext.maxpool1d(x, output, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation)
    
    return output

# Test parameters (used by evaluation framework)
batch_size = 64
features = 192
sequence_length = 65536
kernel_size = 8
stride      = 1
padding     = 4
dilation    = 3            
return_indices = False

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]

def get_inputs():
    x = torch.rand(batch_size, features, sequence_length)
    return [x]
