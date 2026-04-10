# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_071708/code_4.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv1d_weight', 'conv1d_bias', 'conv1d_stride', 'conv1d_padding', 'conv1d_dilation', 'conv1d_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv1d_weight', 'conv1d_bias']


class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool=False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

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
    # State for conv1d (nn.Conv2d)
    if 'conv1d_weight' in flat_state:
        state_kwargs['conv1d_weight'] = flat_state['conv1d_weight']
    else:
        state_kwargs['conv1d_weight'] = getattr(model.conv1d, 'weight', None)
    if 'conv1d_bias' in flat_state:
        state_kwargs['conv1d_bias'] = flat_state['conv1d_bias']
    else:
        state_kwargs['conv1d_bias'] = getattr(model.conv1d, 'bias', None)
    state_kwargs['conv1d_stride'] = model.conv1d.stride
    state_kwargs['conv1d_padding'] = model.conv1d.padding
    state_kwargs['conv1d_dilation'] = model.conv1d.dilation
    state_kwargs['conv1d_groups'] = model.conv1d.groups
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

# CUDA kernel with shared memory weight caching
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv1d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_width,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    extern __shared__ float s_weights[]; // Cache weights here

    int out_ch = blockIdx.z;
    int batch_idx = blockIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;

    // Load weights into shared memory once per block
    if (threadIdx.x < in_channels * kernel_size) {
        s_weights[threadIdx.x] = weight[out_ch * in_channels * kernel_size + threadIdx.x];
    }
    __syncthreads();

    if (out_x < output_width) {
        float sum = (bias != nullptr) ? (float)bias[out_ch] : 0.0f;
        
        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
            for (int k = 0; k < kernel_size; ++k) {
                int input_x = out_x * stride + k * dilation - padding;
                if (input_x >= 0 && input_x < input_width) {
                    float val = (float)input[((batch_idx * in_channels + in_ch) * input_width) + input_x];
                    sum += val * s_weights[in_ch * kernel_size + k];
                }
            }
        }
        output[((batch_idx * out_channels + out_ch) * output_width) + out_x] = (scalar_t)sum;
    }
}

void conv1d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, int stride, int padding, int dilation) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_width = input.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int output_width = output.size(2);

    const int threads = 256;
    const int blocks_x = (output_width + threads - 1) / threads;
    const dim3 grid(blocks_x, batch_size, out_channels);
    
    size_t shared_size = in_channels * kernel_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv1d_kernel", ([&] {
        conv1d_kernel<scalar_t><<<grid, threads, shared_size>>>(
            input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(), batch_size, in_channels, out_channels,
            input_width, output_width, kernel_size, stride, padding, dilation
        );
    }));
}
"""

cpp_source = r"""
void conv1d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                    torch::Tensor output, int stride, int padding, int dilation);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d_forward", &conv1d_forward, "1D Convolution");
}
"""

fused_ext = load_inline(
    name='fused_conv1d', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv1d_weight, conv1d_bias, conv1d_stride, 
                     conv1d_padding, conv1d_dilation, conv1d_groups):
    b, c, h, w = x.shape
    out_c = conv1d_weight.shape[0]
    k = conv1d_weight.shape[2]
    
    stride = conv1d_stride[0] if isinstance(conv1d_stride, tuple) else conv1d_stride
    padding = conv1d_padding[0] if isinstance(conv1d_padding, tuple) else conv1d_padding
    dilation = conv1d_dilation[0] if isinstance(conv1d_dilation, tuple) else conv1d_dilation
    
    out_w = (w + 2 * padding - dilation * (k - 1) - 1) // stride + 1
    output = torch.empty((b, out_c, h, out_w), device=x.device, dtype=x.dtype)
    
    fused_ext.conv1d_forward(
        x.view(b, c, w), conv1d_weight, 
        conv1d_bias if conv1d_bias is not None else torch.tensor([], device=x.device),
        output.view(b, out_c, out_w), stride, padding, dilation
    )
    return output
