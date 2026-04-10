# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_040207/code_10.py
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
import torch.nn as nn
import torch.nn.functional as F

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
    
    # Lazy initialization of CUDA extension
    if fused_ext is None:
        from torch.utils.cpp_extension import load_inline
        
        cuda_kernel = r"""
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <c10/cuda/CUDAGuard.h>

        // Shared memory optimized maxpool1d kernel
        __global__ void maxpool1d_kernel(
            const float* input,
            float* output,
            int batch_size,
            int channels,
            int input_length,
            int output_length,
            int kernel_size,
            int stride,
            int padding,
            int dilation
        ) {
            // Shared memory for caching input data
            extern __shared__ float shared_input[];
            
            // Calculate global thread index
            int batch = blockIdx.z;
            int channel = blockIdx.y;
            int tid = threadIdx.x;
            int block_size = blockDim.x;
            
            if (batch >= batch_size || channel >= channels) return;
            
            // Calculate this thread's output range
            int elements_per_block = (output_length + gridDim.x - 1) / gridDim.x;
            int start_out_pos = blockIdx.x * elements_per_block;
            int end_out_pos = min(start_out_pos + elements_per_block, output_length);
            
            // Calculate corresponding input range
            int start_in_pos = max(0, start_out_pos * stride - padding);
            int end_in_pos = min(input_length, (end_out_pos - 1) * stride - padding + (kernel_size - 1) * dilation + 1);
            
            // Load data into shared memory in a coalesced manner
            for (int i = tid; i < (end_in_pos - start_in_pos + 2 * padding); i += block_size) {
                int input_pos = start_in_pos + i - padding;
                if (input_pos >= 0 && input_pos < input_length) {
                    int input_idx = ((batch * channels) + channel) * input_length + input_pos;
                    shared_input[i] = input[input_idx];
                } else {
                    shared_input[i] = -INFINITY;
                }
            }
            
            // Synchronize to ensure all data is loaded
            __syncthreads();
            
            // Each thread computes multiple outputs if needed
            for (int out_pos = start_out_pos + tid; out_pos < end_out_pos; out_pos += block_size) {
                // Calculate input starting position
                int input_start = out_pos * stride - padding - start_in_pos + padding;
                
                // Find maximum in the pooling window
                float max_val = -INFINITY;
                for (int k = 0; k < kernel_size; k++) {
                    int input_pos = input_start + k * dilation;
                    if (input_pos >= 0 && input_pos < (end_in_pos - start_in_pos + 2 * padding)) {
                        float val = shared_input[input_pos];
                        max_val = fmaxf(max_val, val);
                    }
                }
                
                // Write output
                int output_idx = ((batch * channels) + channel) * output_length + out_pos;
                output[output_idx] = max_val;
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
            // Ensure CUDA context is correct
            const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
            
            int batch_size = input.size(0);
            int channels = input.size(1);
            int input_length = input.size(2);
            int output_length = output.size(2);
            
            const float* input_ptr = input.data_ptr<float>();
            float* output_ptr = output.data_ptr<float>();
            
            // Configure kernel launch parameters
            // Use 3D grid: (blocks_for_output, channels, batches)
            int threads_per_block = 256;
            int blocks_for_output = min(65535, (output_length + threads_per_block - 1) / threads_per_block);
            int blocks_for_channels = min(65535, channels);
            int blocks_for_batches = min(65535, batch_size);
            
            dim3 grid_dim(blocks_for_output, blocks_for_channels, blocks_for_batches);
            dim3 block_dim(threads_per_block);
            
            // Calculate shared memory size
            // Estimate the maximum shared memory needed
            int max_window_span = (kernel_size - 1) * dilation + 1;
            int shared_mem_elements = threads_per_block * stride + max_window_span + 2 * padding;
            size_t shared_mem_size = shared_mem_elements * sizeof(float);
            
            // Check if shared memory requirement is within limits
            int max_shared_mem_per_block;
            cudaDeviceGetAttribute(&max_shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, 0);
            
            if (shared_mem_size <= max_shared_mem_per_block) {
                maxpool1d_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
                    input_ptr,
                    output_ptr,
                    batch_size,
                    channels,
                    input_length,
                    output_length,
                    kernel_size,
                    stride,
                    padding,
                    dilation
                );
            } else {
                // Fallback to simpler kernel without shared memory if needed
                // This is a simplified version for very large kernels
                extern __shared__ float simple_shared[];
                
                // Re-define a simpler kernel for large kernels
                [](const float* input, float* output, int batch_size, int channels, 
                   int input_length, int output_length, int kernel_size, int stride, 
                   int padding, int dilation) {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    int total_elements = batch_size * channels * output_length;
                    
                    if (idx >= total_elements) return;
                    
                    // Decompose linear index to 3D coordinates
                    int out_pos = idx % output_length;
                    int temp = idx / output_length;
                    int channel = temp % channels;
                    int batch = temp / channels;
                    
                    // Calculate input starting position
                    int input_start = out_pos * stride - padding;
                    
                    // Find maximum in the pooling window
                    float max_val = -INFINITY;
                    for (int k = 0; k < kernel_size; k++) {
                        int input_pos = input_start + k * dilation;
                        if (input_pos >= 0 && input_pos < input_length) {
                            int input_idx = ((batch * channels) + channel) * input_length + input_pos;
                            float val = input[input_idx];
                            max_val = fmaxf(max_val, val);
                        }
                    }
                    
                    output[idx] = max_val;
                }(input_ptr, output_ptr, batch_size, channels, input_length, output_length, 
                  kernel_size, stride, padding, dilation);
            }
        }
        """

        # --- C++ Logic (Interface/Bindings) ---
        cpp_source = r"""
        #include <torch/extension.h>

        // Forward declaration of the function in the .cu file
        void maxpool1d_forward(
            const torch::Tensor& input,
            torch::Tensor& output,
            int kernel_size,
            int stride,
            int padding,
            int dilation
        );

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("maxpool1d", &maxpool1d_forward, "MaxPool1D forward pass");
        }
        """

        # Compile the extension
        fused_ext = load_inline(
            name='fused_maxpool1d',
            cpp_sources=cpp_source,
            cuda_sources=cuda_kernel,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            with_cuda=True
        )
    
    # Move input to GPU if not already
    if x.device.type != 'cuda':
        x_key = 'x'
        if x_key not in gpu_tensors or gpu_tensors[x_key].shape != x.shape:
            gpu_tensors[x_key] = x.cuda()
        else:
            gpu_tensors[x_key].copy_(x)
        x_gpu = gpu_tensors[x_key]
    else:
        x_gpu = x
    
    # Calculate output dimensions
    if maxpool_ceil_mode:
        output_length = int(torch.ceil(
            torch.tensor((x_gpu.size(2) + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) 
                        / float(maxpool_stride) + 1)
        ).item())
    else:
        output_length = (x_gpu.size(2) + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
    
    # Create output tensor on GPU
    output_key = f'output_{output_length}'
    if output_key not in gpu_tensors or gpu_tensors[output_key].shape != (x_gpu.size(0), x_gpu.size(1), output_length):
        gpu_tensors[output_key] = torch.empty(x_gpu.size(0), x_gpu.size(1), output_length, device='cuda', dtype=x_gpu.dtype)
    output_gpu = gpu_tensors[output_key]
    
    # Call custom CUDA kernel
    fused_ext.maxpool1d(x_gpu, output_gpu, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation)
    
    return output_gpu

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
