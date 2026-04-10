# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_041646/code_26.py
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

        __global__ void maxpool1d_kernel_optimized(
            const float* __restrict__ input,
            float* __restrict__ output,
            const int channels,
            const int input_length,
            const int output_length,
            const int kernel_size,
            const int stride,
            const int padding,
            const int dilation
        ) {
            // Grid: (output_length, channels, batch)
            // This maps threads to output elements directly, ensuring coalesced global memory writes
            int out_pos = blockIdx.x * blockDim.x + threadIdx.x;
            int channel = blockIdx.y;
            int batch = blockIdx.z;

            if (out_pos >= output_length) return;

            int input_base = (batch * channels + channel) * input_length;
            int output_base = (batch * channels + channel) * output_length;

            int input_start = out_pos * stride - padding;
            
            float max_val = -1e38f; // Representing -INFINITY for float

            #pragma unroll
            for (int k = 0; k < kernel_size; ++k) {
                int input_pos = input_start + k * dilation;
                if (input_pos >= 0 && input_pos < input_length) {
                    float val = input[input_base + input_pos];
                    if (val > max_val) max_val = val;
                }
            }
            output[output_base + out_pos] = max_val;
        }

        void maxpool1d_forward(
            const torch::Tensor& input,
            torch::Tensor& output,
            int kernel_size,
            int stride,
            int padding,
            int dilation
        ) {
            const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
            
            int batch_size = input.size(0);
            int channels = input.size(1);
            int input_length = input.size(2);
            int output_length = output.size(2);
            
            const int threads = 256;
            dim3 block(threads);
            dim3 grid((output_length + threads - 1) / threads, channels, batch_size);
            
            maxpool1d_kernel_optimized<<<grid, block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
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
        void maxpool1d_forward(const torch::Tensor& input, torch::Tensor& output, int kernel_size, int stride, int padding, int dilation);
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("maxpool1d", &maxpool1d_forward, "Optimized MaxPool1D forward");
        }
        """

        fused_ext = load_inline(
            name='fused_maxpool1d',
            cpp_sources=cpp_source,
            cuda_sources=cuda_kernel,
            extra_cuda_cflags=['-O3', '-use_fast_math'],
            with_cuda=True
        )
    
    x_gpu = x.cuda() if x.device.type != 'cuda' else x
    
    # Calculate output dimensions
    if maxpool_ceil_mode:
        output_length = int(torch.ceil(
            torch.tensor((x_gpu.size(2) + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) 
                        / float(maxpool_stride) + 1)
        ).item())
    else:
        output_length = (x_gpu.size(2) + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
    
    output_gpu = torch.empty(x_gpu.size(0), x_gpu.size(1), output_length, device='cuda', dtype=x_gpu.dtype)
    
    fused_ext.maxpool1d(x_gpu, output_gpu, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation)
    
    return output_gpu
