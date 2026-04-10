# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_055939/code_11.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['kernel_size', 'stride', 'padding', 'dilation']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['maxpool_kernel_size', 'maxpool_stride', 'maxpool_padding', 'maxpool_dilation', 'maxpool_ceil_mode', 'maxpool_return_indices']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 2D.
    """

    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(ModelNew, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

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
    # State for maxpool (nn.MaxPool2d)
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void max_pool2d_kernel_shared(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_h, int in_w,
    int out_h, int out_w,
    int k_size, int stride, int padding) {

    // Shared memory: load input tile
    // Block computes 16x16 output region, needs larger input tile
    extern __shared__ char shared_mem[];
    float* input_tile = (float*)shared_mem;
    
    // Tile dimensions for input
    const int TILE_SIZE = 16;
    const int INPUT_TILE_SIZE = TILE_SIZE + k_size - 1;  // e.g., 16 + 3 - 1 = 18
    
    int block_out_h = blockIdx.y * TILE_SIZE;
    int block_out_w = blockIdx.x * TILE_SIZE;
    int batch_channel_idx = blockIdx.z;
    
    int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;
    
    const float* batch_input = input + (batch_channel_idx * in_h * in_w);
    float* batch_output = output + (batch_channel_idx * out_h * out_w);
    
    // Load input tile into shared memory
    // Calculate the top-left corner of the input region needed
    int input_start_h = block_out_h * stride - padding;
    int input_start_w = block_out_w * stride - padding;
    
    // Cooperatively load input_tile by all threads
    for (int idx = thread_idx; idx < INPUT_TILE_SIZE * INPUT_TILE_SIZE; idx += total_threads) {
        int local_i = idx / INPUT_TILE_SIZE;
        int local_j = idx % INPUT_TILE_SIZE;
        
        int global_i = input_start_h + local_i;
        int global_j = input_start_w + local_j;
        
        if (global_i >= 0 && global_i < in_h && global_j >= 0 && global_j < in_w) {
            input_tile[local_i * INPUT_TILE_SIZE + local_j] = 
                batch_input[global_i * in_w + global_j];
        } else {
            input_tile[local_i * INPUT_TILE_SIZE + local_j] = -FLT_MAX;
        }
    }
    __syncthreads();
    
    // Each thread computes one output element
    int local_out_h = threadIdx.y;
    int local_out_w = threadIdx.x;
    int global_out_h = block_out_h + local_out_h;
    int global_out_w = block_out_w + local_out_w;
    
    if (global_out_h >= out_h || global_out_w >= out_w) {
        return;
    }
    
    float max_val = -FLT_MAX;
    
    // Compute max over kernel window using shared memory data
    #pragma unroll
    for (int ki = 0; ki < k_size; ++ki) {
        #pragma unroll
        for (int kj = 0; kj < k_size; ++kj) {
            // Position in shared memory
            int sh_i = local_out_h * stride + ki;
            int sh_j = local_out_w * stride + kj;
            
            if (sh_i < INPUT_TILE_SIZE && sh_j < INPUT_TILE_SIZE) {
                float val = input_tile[sh_i * INPUT_TILE_SIZE + sh_j];
                max_val = fmaxf(max_val, val);
            }
        }
    }
    
    // Write result
    batch_output[global_out_h * out_w + global_out_w] = max_val;
}

void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, 
                        int k_size, int stride, int padding) {
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_h = output.size(2);
    const int out_w = output.size(3);
    
    const int TILE_SIZE = 16;
    const int INPUT_TILE_SIZE = TILE_SIZE + k_size - 1;
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_w + TILE_SIZE - 1) / TILE_SIZE, 
              (out_h + TILE_SIZE - 1) / TILE_SIZE, 
              batch * channels);
    
    size_t shared_mem_size = INPUT_TILE_SIZE * INPUT_TILE_SIZE * sizeof(float);
    
    max_pool2d_kernel_shared<<<grid, block, shared_mem_size>>>(
        input.contiguous().data_ptr<float>(), 
        output.data_ptr<float>(),
        in_h, in_w, out_h, out_w,
        k_size, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, int k_size, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_forward, "Optimized shared memory Max Pool 2D");
}
"""

module = load_inline(
    name='max_pool2d_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    # Handle dilation (not used in this implementation but needed for interface compatibility)
    if maxpool_dilation != 1:
        raise ValueError("Dilation not supported in this optimized implementation")
    
    # Handle ceil mode by adjusting output size calculation if needed
    h_in, w_in = x.shape[2], x.shape[3]
    
    if maxpool_ceil_mode:
        h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size + maxpool_stride - 1) // maxpool_stride + 1
        w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size + maxpool_stride - 1) // maxpool_stride + 1
    else:
        h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
        w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    # Create output tensor
    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out), device=x.device, dtype=x.dtype)
    
    # Call the optimized CUDA kernel
    module.forward(x, output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    
    # Handle return_indices flag
    if maxpool_return_indices:
        # This would require a separate kernel to track indices, which is out of scope here
        # For now, we'll return None as a placeholder
        return output, None
    else:
        return output
