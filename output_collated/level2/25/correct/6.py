# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_083856/code_11.py
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
    const float* __restrict__ x, const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ out, int batch, int in_c, int h, int w, 
    int out_c, int k_size) {
    
    int n = blockIdx.z;
    int oh = blockIdx.y;
    int ow = blockIdx.x;
    
    // Each thread calculates one output channel's convolution result
    extern __shared__ float channel_sums[];
    int oc = threadIdx.x;
    
    if (oc < out_c) {
        float sum = bias[oc];
        for (int ic = 0; ic < in_c; ++ic) {
            for (int kh = 0; kh < k_size; ++kh) {
                for (int kw = 0; kw < k_size; ++kw) {
                    float val = x[((n * in_c + ic) * h + (oh + kh)) * w + (ow + kw)];
                    sum += val * weight[(((oc * in_c + ic) * k_size + kh) * k_size + kw)];
                }
            }
        }
        channel_sums[oc] = sum;
    }
    __syncthreads();

    // Reduction across channels
    if (oc == 0) {
        float min_val = channel_sums[0];
        for (int i = 1; i < out_c; ++i) {
            if (channel_sums[i] < min_val) min_val = channel_sums[i];
        }
        float res = tanhf(tanhf(min_val));
        int out_h = gridDim.y;
        int out_w = gridDim.x;
        out[(n * out_h + oh) * out_w + ow] = res;
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out) {
    const int batch = x.size(0);
    const int in_c = x.size(1);
    const int h = x.size(2);
    const int w = x.size(3);
    const int out_c = weight.size(0);
    const int k_size = weight.size(2);
    const int out_h = h - k_size + 1;
    const int out_w = w - k_size + 1;

    dim3 blocks(out_w, out_h, batch);
    // Use threads per block equal to next power of 2 of out_c for efficient reduction
    int threads_per_block = 64; 
    size_t shared_mem = out_c * sizeof(float);
    
    fused_conv_min_tanh_kernel<<<blocks, threads_per_block, shared_mem>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), batch, in_c, h, w, out_c, k_size);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused conv min tanh forward");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    out_h = x.size(2) - conv_weight.size(2) + 1
    out_w = x.size(3) - conv_weight.size(3) + 1
    out = torch.empty((x.size(0), 1, out_h, out_w), device=x.device)
    fused_ext.fused_op(x, conv_weight, conv_bias, out)
    return out
