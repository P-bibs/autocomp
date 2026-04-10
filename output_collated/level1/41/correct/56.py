# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_042936/code_12.py
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
        #include <climits>
        #include <cfloat>

        __global__ void maxpool1d_coalesced_kernel(
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
            // Calculate global output index
            int idx = blockIdx.x;
            int out_pos = idx % output_length;
            int tmp = idx / output_length;
            int channel = tmp % channels;
            int batch = tmp / channels;

            // Start position in input
            int start = out_pos * stride - padding;
            
            // Shared memory for reduction
            __shared__ float shared_data[256];
            int tid = threadIdx.x;
            
            // Initialize with minimum float value
            float thread_max = -FLT_MAX;
            
            // Each thread loads one element if within kernel size
            if (tid < kernel_size) {
                int in_pos = start + tid * dilation;
                if (in_pos >= 0 && in_pos < input_length) {
                    const float* in_ptr = input + (batch * channels + channel) * input_length;
                    thread_max = __ldg(&in_ptr[in_pos]);
                }
            }
            
            shared_data[tid] = thread_max;
            __syncthreads();
            
            // Reduction in shared memory
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + s]);
                }
                __syncthreads();
            }
            
            // Write result
            if (tid == 0) {
                float* out_ptr = output + (batch * channels + channel) * output_length;
                out_ptr[out_pos] = shared_data[0];
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
            
            // Launch configuration: one block per output element
            int blocks = batch_size * channels * output_length;
            int threads = min(256, ((kernel_size + 31) / 32) * 32); // Round up to nearest multiple of 32, max 256
            
            maxpool1d_coalesced_kernel<<<blocks, threads>>>(
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
            m.def("maxpool1d", &maxpool1d_forward, "MaxPool1D forward (coalesced)");
        }
        """

        fused_ext = load_inline(
            name='maxpool1d_coalesced',
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
