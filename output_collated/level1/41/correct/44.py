# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_041646/code_23.py
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
        cuda_kernel = r"""
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <algorithm>

        __global__ void maxpool1d_kernel(
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
            // Map threads to sequence dimension for coalesced access
            int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (out_idx >= output_length) return;

            // Map block Y to batch and channel
            int batch_channel = blockIdx.y; 
            int batch = batch_channel / channels;
            int channel = batch_channel % channels;

            const float* input_ptr = input + (batch * channels + channel) * input_length;
            float* output_ptr = output + (batch * channels + channel) * output_length;

            int input_start = out_idx * stride - padding;
            float max_val = -3.402823466e+38F; // -FLT_MAX

            // Perform pooling over the window
            for (int k = 0; k < kernel_size; ++k) {
                int input_pos = input_start + k * dilation;
                if (input_pos >= 0 && input_pos < input_length) {
                    float val = input_ptr[input_pos];
                    if (val > max_val) max_val = val;
                }
            }
            output_ptr[out_idx] = max_val;
        }

        void maxpool1d_forward(
            const torch::Tensor& input, 
            torch::Tensor& output, 
            int k, int s, int p, int d
        ) {
            const int batch = input.size(0);
            const int chans = input.size(1);
            const int in_len = input.size(2);
            const int out_len = output.size(2);
            
            dim3 block(256);
            dim3 grid((out_len + 255) / 256, batch * chans);
            
            maxpool1d_kernel<<<grid, block>>>(
                input.data_ptr<float>(), 
                output.data_ptr<float>(),
                chans, in_len, out_len, k, s, p, d
            );
        }
        """
        cpp_source = r"""
        #include <torch/extension.h>
        void maxpool1d_forward(const torch::Tensor& input, torch::Tensor& output, int k, int s, int p, int d);
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("maxpool1d", &maxpool1d_forward, "Coalesced MaxPool1D");
        }
        """
        fused_ext = load_inline(
            name='fused_maxpool1d',
            cpp_sources=cpp_source,
            cuda_sources=cuda_kernel,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            with_cuda=True
        )

    # Compute output dims
    in_len = x.size(2)
    if maxpool_ceil_mode:
        out_len = (in_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1 + maxpool_stride - 1) // maxpool_stride + 1
        # Adjusted slightly to match ceiling logic precisely
        out_len = (in_len + 2 * maxpool_padding - (maxpool_dilation * (maxpool_kernel_size - 1) + 1)) // maxpool_stride + 1
        if (in_len + 2 * maxpool_padding - (maxpool_dilation * (maxpool_kernel_size - 1) + 1)) % maxpool_stride != 0:
             out_len += 1
    else:
        out_len = (in_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
    
    output = torch.empty((x.size(0), x.size(1), out_len), device=x.device, dtype=x.dtype)
    
    # Ensure memory is contiguous for the kernel
    fused_ext.maxpool1d(x.contiguous(), output, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation)
    
    return output
