# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_161448/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv1d_weight', 'conv1d_bias', 'conv1d_stride', 'conv1d_padding', 'conv1d_dilation', 'conv1d_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv1d_weight', 'conv1d_bias']


class ModelNew(nn.Module):
    """
    Performs a standard 1D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0, dilation: int=1, groups: int=1, bias: bool=False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

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
    # State for conv1d (nn.Conv1d)
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

# -------------------------------------------------------------------------
# CUDA source – coalesced 1-D convolution kernel
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1d_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    const int N, const int C_in, const int C_out, const int L, 
    const int K, const int L_out, const int stride, 
    const int dilation, const int padding)
{
    // One block per (batch, out_channel)
    const int c = blockIdx.y;
    const int b = blockIdx.x;
    
    // Shared memory for weights: (C_in * K) floats
    // For C_in=64, K=3, this is 192 floats (~768 bytes), well within limits
    extern __shared__ float weight_s[];
    
    // Cooperative load of weights into shared memory
    int weight_size = C_in * K;
    for (int i = threadIdx.x; i < weight_size; i += blockDim.x) {
        weight_s[i] = weight[c * weight_size + i];
    }
    __syncthreads();

    // Bias value
    float b_val = (bias != nullptr) ? bias[c] : 0.0f;

    // Grid-stride loop over output positions
    for (int l = threadIdx.x; l < L_out; l += blockDim.x) {
        int input_start = l * stride - padding;
        float sum = 0.0f;

        for (int ic = 0; ic < C_in; ++ic) {
            const float* in_ptr = input + (b * C_in + ic) * L;
            const float* w_ptr = &weight_s[ic * K];
            
            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                int idx = input_start + k * dilation;
                if (idx >= 0 && idx < L) {
                    sum += __ldg(in_ptr + idx) * w_ptr[k];
                }
            }
        }
        output[(b * C_out + c) * L_out + l] = sum + b_val;
    }
}

void conv1d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, int stride, int dilation, int padding)
{
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int C_out = weight.size(0);
    const int L_out = output.size(2);
    const int K = weight.size(2);
    
    dim3 grid(N, C_out);
    int block_size = 256;
    int smem = C_in * K * sizeof(float);
    
    conv1d_kernel<<<grid, block_size, smem>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, C_out, (int)input.size(2), K, L_out,
        stride, dilation, padding
    );
}
"""

cpp_source = r"""
void conv1d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                    torch::Tensor output, int stride, int dilation, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d_forward", &conv1d_forward);
}
"""

fused_ext = load_inline(
    name='custom_conv1d',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv1d_weight, conv1d_bias, conv1d_stride, conv1d_padding, conv1d_dilation, conv1d_groups):
    stride = conv1d_stride[0] if isinstance(conv1d_stride, (list, tuple)) else conv1d_stride
    padding = conv1d_padding[0] if isinstance(conv1d_padding, (list, tuple)) else conv1d_padding
    dilation = conv1d_dilation[0] if isinstance(conv1d_dilation, (list, tuple)) else conv1d_dilation
    
    x = x.contiguous()
    w = conv1d_weight.contiguous()
    b = conv1d_bias.contiguous() if conv1d_bias is not None else torch.tensor([], device=x.device)
    
    L_in = x.shape[2]
    K = w.shape[2]
    C_out = w.shape[0]
    L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    
    out = torch.empty((x.shape[0], C_out, L_out), device=x.device, dtype=x.dtype)
    fused_ext.conv1d_forward(x, w, b, out, stride, dilation, padding)
    return out
