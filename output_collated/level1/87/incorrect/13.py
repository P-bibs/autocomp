# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_070444/code_6.py
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

# The custom CUDA kernel implements a basic 2D convolution.
# For simplicity and performance within a single-file constraint, this implementation
# focuses on the core convolution arithmetic.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int C_out, int H, int W,
    int kH, int kW, int stride, int padding) {

    int H_out = (H + 2 * padding - kH) / stride + 1;
    int W_out = (W + 2 * padding - kW) / stride + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C_out * H_out * W_out;

    if (idx < total_elements) {
        int w_out = idx % W_out;
        int h_out = (idx / W_out) % H_out;
        int c_out = (idx / (W_out * H_out)) % C_out;
        int b = idx / (W_out * H_out * C_out);

        float sum = 0.0f;
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int ky = 0; ky < kH; ++ky) {
                for (int kx = 0; kx < kW; ++kx) {
                    int h_in = h_out * stride + ky - padding;
                    int w_in = w_out * stride + kx - padding;
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        sum += input[((b * C_in + c_in) * H + h_in) * W + w_in] * 
                               weight[(((c_out * C_in + c_in) * kH + ky) * kW + kx)];
                    }
                }
            }
        }
        output[idx] = sum + (bias ? bias[c_out] : 0.0f);
    }
}

torch::Tensor fused_conv2d(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    int stride, int padding) {
    
    int B = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int C_out = weight.size(0);
    int kH = weight.size(2);
    int kW = weight.size(3);
    
    int H_out = (H + 2 * padding - kH) / stride + 1;
    int W_out = (W + 2 * padding - kW) / stride + 1;
    
    auto output = torch::zeros({B, C_out, H_out, W_out}, x.options());
    
    int total_elements = B * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.data_ptr<float>(), output.data_ptr<float>(),
        B, C_in, C_out, H, W, kH, kW, stride, padding
    );
    
    return output;
}
"""

cpp_source = r"""
torch::Tensor fused_conv2d(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv2d", &fused_conv2d, "Fused 2D Convolution");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3'],
    with_cuda=True
)

def functional_model(x, *, conv1d_weight, conv1d_bias, conv1d_stride, conv1d_padding, conv1d_dilation, conv1d_groups):
    # Note: Simplified to 2D standard convolution logic. Original used 1D parameter names for a 2D call.
    # We pass the parameters directly to our custom CUDA implementation.
    return fused_ext.fused_conv2d(x, conv1d_weight, conv1d_bias, conv1d_stride, conv1d_padding)

batch_size = 16
in_channels = 64
out_channels = 128
width = 1024
height = 1024

def get_init_inputs():
    return [in_channels, out_channels]

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width, device='cuda')
    return [x]
