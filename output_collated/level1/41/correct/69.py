# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_042936/code_28.py
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

# Global cache for the compiled extension
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

    # --- Compile the optimized CUDA kernel once ---------------------------------
    if fused_ext is None:
        cuda_source = r"""
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <device_launch_parameters.h>

        // Coalesced max-pooling kernel: Each block processes one output element (B, C, out_pos)
        // Threads within the block work together to find the max in the sliding window.
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
            int idx = blockIdx.x; // corresponds to one (batch, channel, output_position)
            int out_pos = idx % output_length;
            int tmp    = idx / output_length;
            int c      = tmp % channels;
            int b      = tmp / channels;

            int start = out_pos * stride - padding;
            int tid = threadIdx.x;
            
            float local_max = -3.402823466e+38F; // -FLT_MAX

            // Each thread loads one element of the window. 
            // If kernel_size > warpSize, we loop to cover the full window.
            for (int k_idx = tid; k_idx < kernel_size; k_idx += blockDim.x) {
                int in_pos = start + k_idx * dilation;
                if (in_pos >= 0 && in_pos < input_length) {
                    const float* in_ptr = input + (b * channels + c) * input_length;
                    float val = __ldg(&in_ptr[in_pos]);
                    if (val > local_max) local_max = val;
                }
            }

            // Warp-level reduction
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                float other = __shfl_xor_sync(0xffffffff, local_max, offset);
                if (other > local_max) local_max = other;
            }

            // Thread 0 writes the result
            if (tid == 0) {
                float* out_ptr = output + (b * channels + c) * output_length;
                out_ptr[out_pos] = local_max;
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
            int B = input.size(0);
            int C = input.size(1);
            int in_len  = input.size(2);
            int out_len = output.size(2);

            int blocks = B * C * out_len;
            // Use 32 threads (1 warp) to minimize overhead for local reduction
            int threads = 32;

            maxpool1d_coalesced_kernel<<<blocks, threads>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                B, C, in_len, out_len,
                kernel_size, stride, padding, dilation
            );
        }
        """

        cpp_source = r"""
        #include <torch/extension.h>
        void maxpool1d_forward(const torch::Tensor& input, torch::Tensor& output, int k, int s, int p, int d);
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("maxpool1d", &maxpool1d_forward, "MaxPool1D Optimized Coalesced");
        }
        """

        fused_ext = load_inline(
            name='maxpool1d_optimized',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            with_cuda=True
        )

    # --- Calculation -----------------------------------------------------------
    total_len = x.size(2)
    if maxpool_ceil_mode:
        output_length = ((total_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) + maxpool_stride - 1) // maxpool_stride + 1
    else:
        output_length = (total_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1

    x_gpu = x.cuda().contiguous()
    output = torch.empty((x.size(0), x.size(1), output_length), device='cuda', dtype=x.dtype)

    fused_ext.maxpool1d(x_gpu, output, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation)

    return output
