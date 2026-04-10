# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_113951/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['kernel_size', 'stride', 'padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['avg_pool_kernel_size', 'avg_pool_stride', 'avg_pool_padding', 'avg_pool_ceil_mode', 'avg_pool_count_include_pad']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs 1D Average Pooling.
    """

    def __init__(self, kernel_size: int, stride: int=1, padding: int=0):
        """
        Initializes the 1D Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

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
    # State for avg_pool (nn.AvgPool1d)
    state_kwargs['avg_pool_kernel_size'] = model.avg_pool.kernel_size
    state_kwargs['avg_pool_stride'] = model.avg_pool.stride
    state_kwargs['avg_pool_padding'] = model.avg_pool.padding
    state_kwargs['avg_pool_ceil_mode'] = model.avg_pool.ceil_mode
    state_kwargs['avg_pool_count_include_pad'] = model.avg_pool.count_include_pad
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

# ----------------------------------------------------------------------
# CUDA kernel + host code (shared-memory tiled average-pooling)
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ------------------------------------------------------------
// Tiled 1-D average-pooling kernel using shared memory
// ------------------------------------------------------------
__global__ void avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch,
    const int channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const bool ceil_mode,
    const bool count_include_pad)
{
    // ----- shared memory for the current tile -----
    // size = tile_width * stride + kernel_size - 1
    extern __shared__ float sdata[];

    // ----- block identification -----
    const int block_id = blockIdx.x;                     // one block per (batch, channel)
    const size_t input_base  = (size_t)block_id * input_length;
    const size_t output_base = (size_t)block_id * output_length;

    // ----- tile geometry -----
    const int tile_width = blockDim.x;                  // number of output positions per block
    const int needed = tile_width * stride + kernel_size - 1;   // elements to load
    const int tile_start = (block_id * tile_width) * stride - padding;

    // ----- load input tile into shared memory -----
    for (int i = threadIdx.x; i < needed; i += blockDim.x) {
        int in_idx = tile_start + i;
        float val = 0.0f;
        if (in_idx >= 0 && in_idx < input_length) {
            val = input[input_base + in_idx];
        }
        // else: padded zero (remains 0)
        sdata[i] = val;
    }
    __syncthreads();

    // ----- compute one output per thread -----
    const int out_i = block_id * tile_width + threadIdx.x;
    if (out_i >= output_length) return;

    // start index of the window in the original tensor
    const int start = out_i * stride - padding;
    // offset inside the shared-memory tile
    const int offset = start - tile_start;   // = threadIdx.x * stride

    float sum = 0.0f;
    int count = 0;
    for (int j = 0; j < kernel_size; ++j) {
        const int idx = offset + j;          // position inside sdata
        const float val = sdata[idx];
        sum += val;

        // determine whether this position corresponds to a real input element
        const int global_idx = tile_start + idx;
        if (global_idx >= 0 && global_idx < input_length) {
            ++count;
        }
    }

    float avg;
    if (count_include_pad) {
        avg = sum / static_cast<float>(kernel_size);
    } else {
        avg = (count > 0) ? sum / static_cast<float>(count) : 0.0f;
    }

    output[output_base + out_i] = avg;
}

// ------------------------------------------------------------
// Host wrapper called from Python
// ------------------------------------------------------------
at::Tensor avg_pool1d_cuda(
    at::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    bool ceil_mode,
    bool count_include_pad)
{
    // Make sure the tensor is contiguous and on the GPU
    auto input_cont = input.contiguous();
    TORCH_CHECK(input_cont.is_cuda(), "Input tensor must be a CUDA tensor");

    const int batch      = input_cont.size(0);
    const int channels   = input_cont.size(1);
    const int in_len     = input_cont.size(2);

    // ----- compute output length -----
    int temp = in_len + 2 * padding - kernel_size;   // (L + 2*pad - k)
    int out_len;
    if (ceil_mode) {
        out_len = (temp + stride - 1) / stride + 1; // ceil
    } else {
        out_len = temp / stride + 1;                // floor
    }

    // Allocate output tensor
    auto output = at::empty({batch, channels, out_len}, input_cont.options());

    // ----- kernel launch parameters -----
    const int block_dim = 128;                      // multiple of 32, good occupancy
    const int grid_dim  = batch * channels;         // one block per (batch, channel)

    const int needed = block_dim * stride + kernel_size - 1;
    const int shared_bytes = needed * sizeof(float);

    avg_pool1d_kernel<<<grid_dim, block_dim, shared_bytes>>>(
        input_cont.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, in_len, out_len,
        kernel_size, stride, padding,
        ceil_mode, count_include_pad
    );

    cudaDeviceSynchronize();   // ensure the kernel finished
    return output;
}
"""

# ------------------------------------------------------------
# C++ binding (PyBind11)
# ------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

at::Tensor avg_pool1d_cuda(
    at::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    bool ceil_mode,
    bool count_include_pad);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d", &avg_pool1d_cuda,
          "Custom 1-D average pooling with shared-memory optimisation");
}
"""

# ------------------------------------------------------------
# Compile the extension
# ------------------------------------------------------------
fused_ext = load_inline(
    name='avg_pool_cuda',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Helper functions required by the original harness
# ----------------------------------------------------------------------
batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4

def get_init_inputs():
    return [kernel_size, stride, padding]

def get_inputs():
    x = torch.rand(batch_size, in_channels, input_length)
    return [x]

# ----------------------------------------------------------------------
# The function that will be evaluated
# ----------------------------------------------------------------------
def functional_model(
    x,
    *,
    avg_pool_kernel_size,
    avg_pool_stride,
    avg_pool_padding,
    avg_pool_ceil_mode,
    avg_pool_count_include_pad
):
    """
    Performs 1-D average pooling using a custom CUDA kernel that
    loads a tile of the input into shared memory before computing
    the output (optimisation #2).
    """
    # Ensure the input is on the GPU
    if not x.is_cuda:
        x = x.cuda()

    # Call the compiled custom kernel
    out = fused_ext.avg_pool1d(
        x,
        avg_pool_kernel_size,
        avg_pool_stride,
        avg_pool_padding,
        avg_pool_ceil_mode,
        avg_pool_count_include_pad
    )
    return out
