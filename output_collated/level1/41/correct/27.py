# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_040207/code_21.py
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

# Global persistent extension
fused_ext = None

def get_ext():
    global fused_ext
    if fused_ext is not None:
        return fused_ext
    
    cuda_kernel = r"""
    #include <torch/extension.h>
    #include <cuda_runtime.h>
    #include <c10/cuda/CUDAGuard.h>
    #include <algorithm>

    __global__ void maxpool1d_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        int total_elements,
        int channels,
        int input_length,
        int output_length,
        int k_size,
        int stride,
        int padding,
        int dilation
    ) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total_elements) return;

        // Map linear index to (batch, channel, out_pos)
        // Order: (batches * channels * output_length)
        int out_pos = tid % output_length;
        int rem = tid / output_length;
        int channel = rem % channels;
        int batch = rem / channels;
        
        int input_base = (batch * channels + channel) * input_length;
        int input_start = out_pos * stride - padding;
        
        float max_val = -3.40282e+38f; // FLT_MIN equivalent
        
        #pragma unroll
        for (int k = 0; k < k_size; ++k) {
            int in_idx = input_start + k * dilation;
            if (in_idx >= 0 && in_idx < input_length) {
                float val = input[input_base + in_idx];
                if (val > max_val) max_val = val;
            }
        }
        output[tid] = max_val;
    }

    void maxpool1d_forward(
        const torch::Tensor& input,
        torch::Tensor& output,
        int k, int s, int p, int d
    ) {
        const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
        int total = output.numel();
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        
        maxpool1d_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            total,
            (int)input.size(1),
            (int)input.size(2),
            (int)output.size(2),
            k, s, p, d
        );
    }
    """

    cpp_source = r"""
    #include <torch/extension.h>
    void maxpool1d_forward(const torch::Tensor&, torch::Tensor&, int, int, int, int);
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("maxpool1d_forward", &maxpool1d_forward, "Maxpool 1D forward");
    }
    """

    fused_ext = load_inline(
        name='maxpool1d_optimized',
        cpp_sources=cpp_source,
        cuda_sources=cuda_kernel,
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        with_cuda=True
    )
    return fused_ext

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    ext = get_ext()
    
    # Ensure input is on GPU (minimal latency if already there)
    if not x.is_cuda:
        x = x.cuda()
    
    # Calculate output dimension
    in_len = x.size(2)
    numer = in_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1
    if maxpool_ceil_mode:
        out_len = (numer + maxpool_stride - 1) // maxpool_stride + 1
    else:
        out_len = numer // maxpool_stride + 1
        
    output = torch.empty((x.size(0), x.size(1), out_len), device=x.device, dtype=x.dtype)
    
    # Launch Kernel
    ext.maxpool1d_forward(x, output, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation)
    
    return output
