# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_035026/code_9.py
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

        #define TILE_SIZE 256
        #define SMEM_PITCH(tile_in_length) ((tile_in_length) + 1)

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
            extern __shared__ float smem[];
            
            int batch_channel_idx = blockIdx.x;
            int batch = batch_channel_idx / channels;
            int channel = batch_channel_idx % channels;
            
            if (batch >= batch_size) return;
            
            int tid = threadIdx.x;
            
            // Calculate input tile dimensions
            int tile_out_length = TILE_SIZE;
            int tile_in_length = (kernel_size - 1) * dilation + tile_out_length;
            int smem_pitch = SMEM_PITCH(tile_in_length);
            
            // Base pointers for this batch and channel
            const float* input_base = input + (batch * channels + channel) * input_length;
            float* output_base = output + (batch * channels + channel) * output_length;
            
            // Process multiple output tiles if needed
            for (int tile_start = 0; tile_start < output_length; tile_start += TILE_SIZE) {
                int actual_tile_size = min(TILE_SIZE, output_length - tile_start);
                
                // Load phase: Each thread loads one element into shared memory
                int global_input_pos = (tile_start + tid) * stride - padding;
                
                if (tid < actual_tile_size) {
                    if (global_input_pos >= 0 && global_input_pos < input_length) {
                        smem[tid] = input_base[global_input_pos];
                    } else {
                        smem[tid] = -INFINITY;
                    }
                }
                
                // Additional loads for extended window
                int extra_loads = (tile_in_length + TILE_SIZE - 1) / TILE_SIZE;
                for (int i = 1; i < extra_loads; i++) {
                    int load_idx = tid + i * TILE_SIZE;
                    if (load_idx < tile_in_length && tid < actual_tile_size) {
                        int global_pos = (tile_start + load_idx) * stride - padding;
                        if (global_pos >= 0 && global_pos < input_length) {
                            smem[load_idx] = input_base[global_pos];
                        } else {
                            smem[load_idx] = -INFINITY;
                        }
                    }
                }
                
                __syncthreads();
                
                // Reduction phase: Each thread computes max for its output position
                if (tid < actual_tile_size) {
                    float local_max = -INFINITY;
                    int out_pos = tile_start + tid;
                    
                    // Perform reduction over kernel window
                    for (int k = 0; k < kernel_size; k++) {
                        int smem_pos = tid + k * dilation;
                        if (smem_pos < tile_in_length) {
                            local_max = fmaxf(local_max, smem[smem_pos]);
                        }
                    }
                    
                    output_base[out_pos] = local_max;
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
            // Ensure CUDA context is correct
            const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
            
            int batch_size = input.size(0);
            int channels = input.size(1);
            int input_length = input.size(2);
            int output_length = output.size(2);
            
            const float* input_ptr = input.data_ptr<float>();
            float* output_ptr = output.data_ptr<float>();
            
            // Calculate grid and block dimensions
            int blocks = (batch_size * channels + TILE_SIZE - 1) / TILE_SIZE;
            int threads = TILE_SIZE;
            
            // Calculate shared memory size
            int tile_in_length = (kernel_size - 1) * dilation + TILE_SIZE;
            int smem_pitch = SMEM_PITCH(tile_in_length);
            size_t smem_size = smem_pitch * sizeof(float);
            
            maxpool1d_kernel<<<blocks, threads, smem_size>>>(
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
