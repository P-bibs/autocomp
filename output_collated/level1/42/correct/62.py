# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_055939/code_14.py
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
#include <c10/cuda/CUDAGuard.h>

// Tile size matches the original 16×16 output tile
constexpr int TILE_SIZE = 16;

__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w,
    const int k_size,
    const int stride,
    const int padding) {

    // Shared memory for the tile plus halo
    extern __shared__ float shared_mem[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;                     // flattened (batch, channel) index

    // Output coordinates owned by this thread
    const int ow = bx * TILE_SIZE + tx;
    const int oh = by * TILE_SIZE + ty;

    // Starting input position for this block's region
    const int iw_start = bx * TILE_SIZE * stride - padding;
    const int ih_start = by * TILE_SIZE * stride - padding;

    // Size of the shared-memory region we must fill
    const int shared_width  = TILE_SIZE * stride + k_size - 1;
    const int shared_height = TILE_SIZE * stride + k_size - 1;

    // -------------------------------------------------
    //  Coalesced load of the required input region
    //  Each thread handles several positions, stepping
    //  by the block dimension in both dimensions.
    // -------------------------------------------------
    for (int i = ty; i < shared_height; i += blockDim.y) {
        int ih = ih_start + i;
        for (int j = tx; j < shared_width; j += blockDim.x) {
            int iw = iw_start + j;
            float val;
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                // read-only cache load
                val = __ldg(&input[(bz * in_h + ih) * in_w + iw]);
            } else {
                val = -1e38f;                     // padding value
            }
            shared_mem[i * shared_width + j] = val;
        }
    }

    __syncthreads();

    // -------------------------------------------------
    //  Compute max pooling for the output position
    // -------------------------------------------------
    if (ow < out_w && oh < out_h) {
        float max_val = -1e38f;
        const int local_ih_start = ty * stride;
        const int local_iw_start = tx * stride;

        #pragma unroll
        for (int di = 0; di < k_size; ++di) {
            const int sh_i = local_ih_start + di;
            #pragma unroll
            for (int dj = 0; dj < k_size; ++dj) {
                const int sh_j = local_iw_start + dj;
                float val = shared_mem[sh_i * shared_width + sh_j];
                if (val > max_val) max_val = val;
            }
        }

        const int out_idx = ((bz * out_h + oh) * out_w + ow);
        output[out_idx] = max_val;
    }
}

void max_pool2d_forward(const torch::Tensor& input,
                        torch::Tensor& output,
                        const int k_size,
                        const int stride,
                        const int padding) {
    const int batch_size = input.size(0);
    const int channels   = input.size(1);
    const int in_h       = input.size(2);
    const int in_w       = input.size(3);
    const int out_h      = output.size(2);
    const int out_w      = output.size(3);

    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid((out_w + TILE_SIZE - 1) / TILE_SIZE,
                    (out_h + TILE_SIZE - 1) / TILE_SIZE,
                    batch_size * channels);

    const int shared_width  = TILE_SIZE * stride + k_size - 1;
    const int shared_height = TILE_SIZE * stride + k_size - 1;
    const size_t shared_mem = shared_width * shared_height * sizeof(float);

    // Ensure we're on the correct device
    at::cuda::CUDAGuard device_guard(input.device());
    
    max_pool2d_kernel<<<grid, block, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, in_h, in_w, out_h, out_w,
        k_size, stride, padding
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool2d_forward(const torch::Tensor& input,
                        torch::Tensor& output,
                        const int k_size,
                        const int stride,
                        const int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_forward, "Max pool 2D forward (coalesced version)");
}
"""

# Compile the extension
module = load_inline(
    name='max_pool2d_coalesced',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding,
                     maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    # Compute output spatial size (standard formula, ignoring ceil_mode)
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1

    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out),
                         device=x.device, dtype=x.dtype)

    # Ensure input tensor is contiguous
    x_contiguous = x.contiguous()
    
    # Invoke the optimized CUDA kernel
    module.forward(x_contiguous, output,
                   maxpool_kernel_size, maxpool_stride, maxpool_padding)

    if maxpool_return_indices:
        raise NotImplementedError("Indices are not supported in this optimized kernel.")
    return output
