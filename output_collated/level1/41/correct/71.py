# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_044333/code_4.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <float.h>

__global__ void maxpool1d_kernel_vec(
    const float* __restrict__ input,
    float* __restrict__ output,
    int channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int batch_idx = blockIdx.y;
    int channel_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // Pointer arithmetic
    int input_offset = (batch_idx * channels + channel_idx) * input_length;
    int output_offset = (batch_idx * channels + channel_idx) * output_length;
    const float* in_ptr = input + input_offset;
    float* out_ptr = output + output_offset;
    
    for (int out_pos = tid; out_pos < output_length; out_pos += blockDim.x) {
        int start_pos = out_pos * stride - padding;
        float max_val = -FLT_MAX;
        
        // Process 4 elements at a time using float4 where possible
        int k = 0;
        if (dilation == 1) {
            // Vectorized processing when no dilation
            for (; k + 4 <= kernel_size; k += 4) {
                int in_pos = start_pos + k;
                if (in_pos >= 0 && in_pos + 3 < input_length) {
                    float4 vals = reinterpret_cast<const float4*>(&in_ptr[in_pos])[0];
                    max_val = fmaxf(max_val, fmaxf(fmaxf(vals.x, vals.y), fmaxf(vals.z, vals.w)));
                } else {
                    // Handle boundary conditions element by element
                    for (int i = 0; i < 4; i++) {
                        int pos = in_pos + i;
                        if (pos >= 0 && pos < input_length) {
                            max_val = fmaxf(max_val, in_ptr[pos]);
                        }
                    }
                }
            }
        }
        
        // Process remaining elements
        for (; k < kernel_size; ++k) {
            int in_pos = start_pos + k * dilation;
            if (in_pos >= 0 && in_pos < input_length) {
                max_val = fmaxf(max_val, in_ptr[in_pos]);
            }
        }
        
        out_ptr[out_pos] = max_val;
    }
}

void maxpool1d_forward(const torch::Tensor& input, torch::Tensor& output, int k, int s, int p, int d) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int output_length = output.size(2);
    
    dim3 grid(channels, batch_size);
    int threads = min(output_length, 256);
    threads = (threads == 0) ? 1 : threads;
    
    maxpool1d_kernel_vec<<<grid, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        channels, input.size(2), output_length, k, s, p, d);
}
"""

cpp_source = r"""
void maxpool1d_forward(const torch::Tensor& input, torch::Tensor& output, int k, int s, int p, int d);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool1d", &maxpool1d_forward, "Vectorized MaxPool1D forward");
}
"""

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    global fused_ext
    if fused_ext is None:
        fused_ext = load_inline(name='maxpool1d_vec', cpp_sources=cpp_source, cuda_sources=cuda_source, 
                                extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'], with_cuda=True)
    
    total_len = x.size(2)
    k, s, p, d = maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation
    
    if maxpool_ceil_mode:
        output_length = ((total_len + 2 * p - d * (k - 1) - 1) + s - 1) // s + 1
    else:
        output_length = (total_len + 2 * p - d * (k - 1) - 1) // s + 1
    
    x_gpu = x.contiguous().cuda()
    output = torch.empty((x.size(0), x.size(1), output_length), device='cuda', dtype=x.dtype)
    fused_ext.maxpool1d(x_gpu, output, k, s, p, d)
    return output
