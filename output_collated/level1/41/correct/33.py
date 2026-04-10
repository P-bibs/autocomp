# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_041646/code_5.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Vectorized MaxPool1D kernel using float4
__global__ void maxpool1d_kernel_vec(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int features,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_and_batch = blockIdx.y;
    
    if (tid >= output_length) return;

    int batch_idx = feature_and_batch / features;
    int feat_idx = feature_and_batch % features;

    int input_offset = (batch_idx * features + feat_idx) * input_length;
    int output_offset = (batch_idx * features + feat_idx) * output_length + tid;

    int start_pos = tid * stride - padding;
    float max_val = -3.40282e38f; // FLT_MIN

    // Handle vectorized loads when possible (stride=1, no dilation effects on alignment)
    if (stride == 1 && dilation == 1 && (kernel_size % 4 == 0) && (start_pos % 4 == 0)) {
        const float4* input4 = reinterpret_cast<const float4*>(input + input_offset);
        int start_pos4 = start_pos / 4;
        int kernel_size4 = kernel_size / 4;
        
        #pragma unroll 4
        for (int k = 0; k < kernel_size4; ++k) {
            int pos4 = start_pos4 + k;
            if (pos4 >= 0 && pos4 < (input_length / 4)) {
                float4 val4 = input4[pos4];
                if (val4.x > max_val) max_val = val4.x;
                if (val4.y > max_val) max_val = val4.y;
                if (val4.z > max_val) max_val = val4.z;
                if (val4.w > max_val) max_val = val4.w;
            }
        }
    } else {
        // Fallback to element-wise processing
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            int pos = start_pos + k * dilation;
            if (pos >= 0 && pos < input_length) {
                float val = input[input_offset + pos];
                if (val > max_val) max_val = val;
            }
        }
    }
    
    output[output_offset] = max_val;
}

void maxpool1d_cuda(
    torch::Tensor input,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int batch_size = input.size(0);
    int features = input.size(1);
    int input_length = input.size(2);
    int output_length = output.size(2);

    dim3 threads(256);
    dim3 blocks((output_length + 255) / 256, batch_size * features);

    maxpool1d_kernel_vec<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        features,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation
    );
}
"""

cpp_source = r"""
void maxpool1d_cuda(torch::Tensor input, torch::Tensor output, int kernel_size, int stride, int padding, int dilation);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool1d", &maxpool1d_cuda, "Vectorized MaxPool1d CUDA");
}
"""

# Compile extension
maxpool_ext = load_inline(
    name='maxpool_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

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
    if maxpool_return_indices:
        raise NotImplementedError("Indices not supported in custom kernel")
    
    x = x.contiguous()
    batch, feat, seq_len = x.shape
    
    # Standard formula for MaxPool output size
    if maxpool_ceil_mode:
        out_len = (seq_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1 + maxpool_stride - 1) // maxpool_stride + 1
    else:
        out_len = (seq_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
        
    output = torch.empty((batch, feat, out_len), device=x.device, dtype=x.dtype)
    
    maxpool_ext.maxpool1d(
        x, output, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation
    )
    
    return output

# Inputs/Constants preserved for compatibility
batch_size = 64
features = 192
sequence_length = 65536
kernel_size = 8
stride = 1
padding = 4
dilation = 3
return_indices = False

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]

def get_inputs():
    return [torch.rand(batch_size, features, sequence_length).cuda()]
