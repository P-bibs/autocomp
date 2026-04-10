# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_120739/code_3.py
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
# CUDA source – a custom 1-D average-pooling kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const bool count_include_pad,
    const bool ceil_mode
) {
    // Shared memory for the tile of input needed by this block
    extern __shared__ float s_data[];

    const int tid = threadIdx.x;
    const int n = blockIdx.x;                     // index over batch*channels
    const int tile_start = blockIdx.y * blockDim.x;
    const int out_i = tile_start + tid;           // output position handled by this thread

    // Starting input index for the whole tile
    const int in_start = tile_start * stride - padding;
    const int total_loads = blockDim.x + kernel_size - 1;

    // ---- Cooperative load of the tile into shared memory ----
    for (int load_idx = tid; load_idx < total_loads; load_idx += blockDim.x) {
        int gIdx = in_start + load_idx;
        float val = 0.0f;
        if (gIdx >= 0 && gIdx < input_length) {
            // coalesced read from global memory
            val = input[n * input_length + gIdx];
        }
        s_data[load_idx] = val;
    }

    __syncthreads();

    if (out_i >= output_length) return;

    // ---- Compute the sum over the kernel window ----
    float sum = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        sum += s_data[tid + k];
    }

    // ---- Determine denominator (how many elements really contributed) ----
    float denom;
    if (count_include_pad) {
        denom = static_cast<float>(kernel_size);
    } else {
        int left_pad  = (out_i * stride < padding) ? (padding - out_i * stride) : 0;
        int right_pad = ((out_i * stride + kernel_size - 1) > (input_length - 1 + padding))
                        ? ((out_i * stride + kernel_size - 1) - (input_length - 1 + padding)) : 0;
        int valid = kernel_size - left_pad - right_pad;
        if (valid <= 0) {
            denom = 1.0f;               // avoid div-by-zero; result will be zero anyway
        } else {
            denom = static_cast<float>(valid);
        }
    }

    float result = sum / denom;
    output[n * output_length + out_i] = result;
}

void fused_op_forward(
    at::Tensor input,
    at::Tensor output,
    int batch_size,
    int channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    bool count_include_pad,
    bool ceil_mode
) {
    const int blockDim = 1024;                       // 1024 threads per block
    int tile_count = (output_length + blockDim - 1) / blockDim;
    dim3 grid(batch_size * channels, tile_count);
    int shared_mem = (blockDim + kernel_size - 1) * sizeof(float);

    avg_pool1d_kernel<<<grid, blockDim, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        count_include_pad,
        ceil_mode
    );
    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
# C++ bindings – expose the kernel to Python
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    at::Tensor input,
    at::Tensor output,
    int batch_size,
    int channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    bool count_include_pad,
    bool ceil_mode
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused average-pooling forward");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Benchmark helpers (required by the harness, not used by functional_model)
# ----------------------------------------------------------------------
def get_init_inputs():
    return [8, 1, 4]          # kernel_size, stride, padding

def get_inputs():
    batch_size = 64
    in_channels = 128
    input_length = 65536
    x = torch.rand(batch_size, in_channels, input_length,
                   device='cuda', dtype=torch.float32)
    return [x]

# ----------------------------------------------------------------------
# Optimized functional_model – replaces PyTorch's F.avg_pool1d
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
    # Make sure the input lives on the GPU
    if not x.is_cuda:
        x = x.cuda()

    # Compute output length exactly as PyTorch does
    if avg_pool_ceil_mode:
        out_len = (x.shape[2] + 2 * avg_pool_padding - avg_pool_kernel_size +
                   avg_pool_stride - 1) // avg_pool_stride + 1
    else:
        out_len = (x.shape[2] + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1

    # Allocate output tensor
    out = torch.empty((x.shape[0], x.shape[1], out_len),
                      dtype=x.dtype, device=x.device)

    # Launch the custom CUDA kernel
    fused_ext.fused_op(
        x.contiguous(),
        out,
        x.shape[0],          # batch size
        x.shape[1],          # number of channels
        x.shape[2],          # input length
        out_len,             # output length
        avg_pool_kernel_size,
        avg_pool_stride,
        avg_pool_padding,
        avg_pool_count_include_pad,
        avg_pool_ceil_mode
    )
    return out
