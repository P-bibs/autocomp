# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_035026/code_21.py
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

# Global variables for extension and caching
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
    
    # Lazy initialization
    if fused_ext is None:
        cuda_kernel = r"""
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <c10/cuda/CUDAGuard.h>
        #include <algorithm>

        #define BLOCK_DIM 256

        __global__ void maxpool1d_tiled_kernel(
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
            // Each block processes a single (batch, channel) strip
            int total_stripes = batch_size * channels;
            int stripe_idx = blockIdx.x;
            if (stripe_idx >= total_stripes) return;

            extern __shared__ float smem[];

            int batch = stripe_idx / channels;
            int ch = stripe_idx % channels;

            // Shared memory layout: [tile_in_length + padding]
            int tile_out_len = min(BLOCK_DIM, output_length - (blockIdx.y * BLOCK_DIM));
            int tile_in_len = (kernel_size - 1) * dilation + tile_out_len;
            
            // Coalesced loading into shared memory
            // Each thread loads elements to cover the input spatial range
            for (int i = threadIdx.x; i < tile_in_len; i += BLOCK_DIM) {
                int pos = (blockIdx.y * BLOCK_DIM * stride) - padding + i;
                if (pos >= 0 && pos < input_length) {
                    smem[i] = input[((batch * channels) + ch) * input_length + pos];
                } else {
                    smem[i] = -1e38f; // Represent -inf
                }
            }
            __syncthreads();

            // Perform reduction
            int out_idx = blockIdx.y * BLOCK_DIM + threadIdx.x;
            if (out_idx < output_length) {
                float max_val = -1e38f;
                int start_in = threadIdx.x * stride;
                for (int k = 0; k < kernel_size; k++) {
                    max_val = fmaxf(max_val, smem[start_in + k * dilation]);
                }
                output[((batch * channels) + ch) * output_length + out_idx] = max_val;
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
            const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
            int batch_size = input.size(0);
            int channels = input.size(1);
            int input_length = input.size(2);
            int output_length = output.size(2);

            // Grid: (batch*channels, ceil(output_length/BLOCK_DIM))
            dim3 grid(batch_size * channels, (output_length + BLOCK_DIM - 1) / BLOCK_DIM);
            int smem_size = ((kernel_size - 1) * dilation + BLOCK_DIM) * sizeof(float);
            
            maxpool1d_tiled_kernel<<<grid, BLOCK_DIM, smem_size>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size, channels, input_length, output_length,
                kernel_size, stride, padding, dilation
            );
        }
        """

        cpp_source = r"""
        #include <torch/extension.h>
        void maxpool1d_forward(const torch::Tensor& input, torch::Tensor& output, int k, int s, int p, int d);
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("maxpool1d", &maxpool1d_forward, "Tiled MaxPool1D forward");
        }
        """

        fused_ext = load_inline(
            name='fused_maxpool1d',
            cpp_sources=cpp_source,
            cuda_sources=cuda_kernel,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            with_cuda=True
        )

    # Prepare input
    x_gpu = x.cuda() if x.device.type != 'cuda' else x
    
    # Calculate output dims
    val = (x_gpu.size(2) + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1)
    if maxpool_ceil_mode:
        output_length = (val + maxpool_stride - 1) // maxpool_stride + 1
    else:
        output_length = val // maxpool_stride + 1
    
    output_gpu = torch.empty(x_gpu.size(0), x_gpu.size(1), output_length, device='cuda', dtype=x_gpu.dtype)
    
    # Execute
    fused_ext.maxpool1d(x_gpu, output_gpu, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation)
    
    return output_gpu
