# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_091623/code_11.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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
    # State for conv (nn.Conv2d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
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

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weights, 
    const float* __restrict__ bias,
    float* __restrict__ output, 
    int N, int C_in, int C_out, int H, int W, int H_out, int W_out, int K) {
    
    int n = blockIdx.z;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h_out >= H_out || w_out >= W_out) return;
    
    float min_val = 1e30f;
    
    for (int co = 0; co < C_out; ++co) {
        float sum = bias[co];
        const float* weight_ptr = weights + co * C_in * K * K;
        
        for (int ci = 0; ci < C_in; ++ci) {
            const float* in_ptr = input + (n * C_in + ci) * H * W + h_out * W + w_out;
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    sum += in_ptr[kh * W + kw] * weight_ptr[(ci * K + kh) * K + kw];
                }
            }
        }
        float val = tanhf(tanhf(sum));
        if (val < min_val) min_val = val;
    }
    output[n * H_out * W_out + h_out * W_out + w_out] = min_val;
}

void fused_op(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    int N = input.size(0); int C_in = input.size(1);
    int H = input.size(2); int W = input.size(3);
    int C_out = weight.size(0); int K = weight.size(2);
    int H_out = H - K + 1; int W_out = W - K + 1;
    
    dim3 threads(16, 16);
    dim3 grid((W_out + 15) / 16, (H_out + 15) / 16, N);
    
    fused_conv_min_tanh_kernel<<<grid, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C_in, C_out, H, W, H_out, W_out, K);
}
"""

cpp_source = r"""
void fused_op(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused Conv Min Tanh kernel");
}
"""

fused_lib = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1):
    # Constraint enforcement: custom convolution implementation
    batch_size, _, height, width = x.shape
    out_channels = conv_weight.shape[0]
    kernel_size = conv_weight.shape[2]
    out_h, out_w = height - kernel_size + 1, width - kernel_size + 1
    
    output = torch.empty((batch_size, 1, out_h, out_w), device=x.device)
    fused_lib.fused_op(x.contiguous(), conv_weight.contiguous(), conv_bias.contiguous(), output)
    return output
