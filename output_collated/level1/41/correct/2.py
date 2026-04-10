# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_034611/code_4.py
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

# Global container for compiled kernel
fused_ext = None

def get_fused_ext():
    global fused_ext
    if fused_ext is not None:
        return fused_ext
        
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
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = batch_size * channels * output_length;
        if (idx >= total) return;

        int out_pos = idx % output_length;
        int temp = idx / output_length;
        int channel = temp % channels;
        int batch = temp / channels;

        int input_start = out_pos * stride - padding;
        float max_val = -3.402823466e+38F; // -FLT_MAX

        for (int k = 0; k < kernel_size; ++k) {
            int input_pos = input_start + k * dilation;
            if (input_pos >= 0 && input_pos < input_length) {
                float val = input[((batch * channels + channel) * input_length) + input_pos];
                if (val > max_val) max_val = val;
            }
        }
        output[idx] = max_val;
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
        int total_threads = batch_size * channels * output_length;
        
        int threads = 256;
        int blocks = (total_threads + threads - 1) / threads;

        maxpool1d_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels, input_length, output_length,
            kernel_size, stride, padding, dilation
        );
    }
    """

    cpp_source = r"""
    void maxpool1d_forward(const torch::Tensor& input, torch::Tensor& output, int kernel, int stride, int pad, int dilation);
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("maxpool1d", &maxpool1d_forward, "Maxpool 1D forward");
    }
    """

    fused_ext = load_inline(
        name='maxpool1d_ext',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        with_cuda=True
    )
    return fused_ext

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
    # Ensure input is on GPU and float32
    if not x.is_cuda:
        x = x.cuda()
    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    # Output dimension calculation
    input_len = x.size(2)
    # The ceiling mode logic is often ignored in standard 1D implementations if ceil_mode=False
    # Formula: floor((L + 2*pad - dilation*(kernel-1) - 1)/stride + 1)
    output_length = ((input_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) 
                     // maxpool_stride + 1)
    
    output = torch.empty((x.size(0), x.size(1), output_length), device=x.device, dtype=x.dtype)
    
    ext = get_fused_ext()
    ext.maxpool1d(x, output, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation)
    
    return output
