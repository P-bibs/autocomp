# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_054839/code_9.py
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

__global__ void max_pool2d_kernel_shared(
    const float* __restrict__ input,
    float* __restrict__ output,
    int channels, int in_h, int in_w,
    int out_h, int out_w,
    int batch_size_channels,
    int k_size, int stride, int padding) {

    // Grid: (out_w/2, out_h/2, batch * channels)
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z; 

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int ow = bx * blockDim.x * 2;  // Starting output column
    int oh = by * blockDim.y * 2;  // Starting output row

    // Shared memory for input patch
    // Dimensions: (blockDim.y * 2 * stride + k_size) x (blockDim.x * 2 * stride + k_size)
    extern __shared__ float shared_input[];
    
    int shared_h = blockDim.y * 2 * stride + k_size;
    int shared_w = blockDim.x * 2 * stride + k_size;
    
    const float* input_ptr = input + (bz * in_h * in_w);
    
    // Cooperative load: threads load input patch into shared memory
    int total_threads = blockDim.x * blockDim.y;
    int patch_size = shared_h * shared_w;
    
    for (int idx = ty * blockDim.x + tx; idx < patch_size; idx += total_threads) {
        int si = idx / shared_w;
        int sj = idx % shared_w;
        
        // Map shared memory position to input position
        int ih = (oh / stride) * stride + si - padding;
        int iw = (ow / stride) * stride + sj - padding;
        
        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            shared_input[si * shared_w + sj] = input_ptr[ih * in_w + iw];
        } else {
            shared_input[si * shared_w + sj] = -FLT_MAX;
        }
    }
    
    __syncthreads();
    
    // Process 2x2 output tile using shared memory
    float max_vals[2][2];
    #pragma unroll
    for(int i = 0; i < 2; ++i) {
        #pragma unroll
        for(int j = 0; j < 2; ++j) {
            max_vals[i][j] = -FLT_MAX;
        }
    }
    
    // Calculate positions in output and corresponding positions in shared memory
    #pragma unroll
    for (int out_i = 0; out_i < 2; ++out_i) {
        #pragma unroll
        for (int out_j = 0; out_j < 2; ++out_j) {
            int oh_tile = oh + out_i * blockDim.y + ty;
            int ow_tile = ow + out_j * blockDim.x + tx;
            
            if (oh_tile >= out_h || ow_tile >= out_w) continue;
            
            // Calculate input region for this output element
            int ih_start = oh_tile * stride - padding;
            int iw_start = ow_tile * stride - padding;
            
            // Calculate corresponding shared memory offset
            int sh_start = ih_start - ((oh / stride) * stride - padding);
            int sw_start = iw_start - ((ow / stride) * stride - padding);
            
            // Max pool within kernel
            #pragma unroll
            for (int ki = 0; ki < k_size; ++ki) {
                #pragma unroll
                for (int kj = 0; kj < k_size; ++kj) {
                    int sh = sh_start + ki;
                    int sw = sw_start + kj;
                    
                    if (sh >= 0 && sh < shared_h && sw >= 0 && sw < shared_w) {
                        float val = shared_input[sh * shared_w + sw];
                        if (val > max_vals[out_i][out_j]) {
                            max_vals[out_i][out_j] = val;
                        }
                    }
                }
            }
        }
    }
    
    __syncthreads();
    
    // Write results
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int oh_tile = oh + i * blockDim.y + ty;
            int ow_tile = ow + j * blockDim.x + tx;
            
            if (oh_tile < out_h && ow_tile < out_w) {
                output[(bz * out_h + oh_tile) * out_w + ow_tile] = max_vals[i][j];
            }
        }
    }
}

void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, 
                        int k_size, int stride, int padding) {
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_h = output.size(2);
    const int out_w = output.size(3);

    dim3 block(16, 16);
    dim3 grid((out_w/2 + block.x - 1) / block.x, (out_h/2 + block.y - 1) / block.y, batch * channels);

    int shared_h = block.y * 2 * stride + k_size;
    int shared_w = block.x * 2 * stride + k_size;
    int shared_mem_size = shared_h * shared_w * sizeof(float);

    max_pool2d_kernel_shared<<<grid, block, shared_mem_size>>>(
        input.contiguous().data_ptr<float>(), output.data_ptr<float>(),
        channels, in_h, in_w, out_h, out_w, batch * channels,
        k_size, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, int k_size, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_forward, "Shared memory optimized Max Pool 2D");
}
"""

module = load_inline(
    name='max_pool2d_shared',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out), device=x.device, dtype=x.dtype)
    module.forward(x, output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    return output
