# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_040207/code_26.py
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

# Global to cache the compiled extension
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
        cuda_src = r"""
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <algorithm>

        #define BLOCK_SIZE 256

        __global__ void maxpool1d_shared_kernel(
            const float* __restrict__ input,
            float* __restrict__ output,
            int batch_size,
            int channels,
            int input_len,
            int output_len,
            int k_size,
            int stride,
            int padding,
            int dilation
        ) {
            int tid = threadIdx.x;
            int b_c_idx = blockIdx.y; // batch * channels
            
            // Shared memory buffer to load input segment
            // Roughly enough to cover the window reach
            extern __shared__ float s_data[];

            int n_elements = batch_size * channels * output_len;
            int g_idx = blockIdx.x * BLOCK_SIZE + tid;

            if (g_idx >= n_elements) return;

            int batch = g_idx / (channels * output_len);
            int channel = (g_idx / output_len) % channels;
            int out_pos = g_idx % output_len;

            int input_start = out_pos * stride - padding;
            float max_val = -1e38f; // Equivalent to -infinity for float

            // Simple coalesced-friendly window search
            // For production, tiling into shared memory provides bigger wins 
            // when kernel_size is large and memory is limited.
            for (int k = 0; k < k_size; ++k) {
                int pos = input_start + k * dilation;
                if (pos >= 0 && pos < input_len) {
                    float val = input[((batch * channels) + channel) * input_len + pos];
                    if (val > max_val) max_val = val;
                }
            }
            output[g_idx] = max_val;
        }

        void maxpool1d_forward(
            const torch::Tensor& input,
            torch::Tensor& output,
            int k_size, int stride, int padding, int dilation
        ) {
            int batch_size = input.size(0);
            int channels = input.size(1);
            int input_len = input.size(2);
            int output_len = output.size(2);

            int total_elements = batch_size * channels * output_len;
            int blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

            maxpool1d_shared_kernel<<<blocks, BLOCK_SIZE>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size, channels, input_len, output_len,
                k_size, stride, padding, dilation
            );
        }
        """
        cpp_src = r"""
        void maxpool1d_forward(const torch::Tensor& input, torch::Tensor& output, int k_size, int stride, int padding, int dilation);
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("maxpool1d", &maxpool1d_forward, "MaxPool1D Optimized");
        }
        """
        fused_ext = load_inline(
            name='fused_maxpool1d',
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            with_cuda=True
        )

    # Calculate output dimensions
    denom = float(maxpool_stride)
    val = (x.size(2) + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) / denom + 1
    output_length = int(torch.ceil(torch.tensor(val)).item()) if maxpool_ceil_mode else int(val)
    
    output = torch.empty(x.size(0), x.size(1), output_length, device=x.device, dtype=x.dtype)
    
    # Ensure input is contiguous for coalesced access
    x_contig = x.contiguous()
    fused_ext.maxpool1d(x_contig, output, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation)
    
    return output
