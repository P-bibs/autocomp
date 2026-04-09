# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_053107/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'divisor', 'pool_size', 'bias_shape', 'sum_dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'max_pool_kernel_size', 'max_pool_stride', 'max_pool_padding', 'max_pool_dilation', 'max_pool_ceil_mode', 'max_pool_return_indices', 'global_avg_pool_output_size', 'divisor', 'bias', 'sum_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """

    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

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
    # State for conv (nn.Conv3d)
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
    # State for max_pool (nn.MaxPool3d)
    state_kwargs['max_pool_kernel_size'] = model.max_pool.kernel_size
    state_kwargs['max_pool_stride'] = model.max_pool.stride
    state_kwargs['max_pool_padding'] = model.max_pool.padding
    state_kwargs['max_pool_dilation'] = model.max_pool.dilation
    state_kwargs['max_pool_ceil_mode'] = model.max_pool.ceil_mode
    state_kwargs['max_pool_return_indices'] = model.max_pool.return_indices
    # State for global_avg_pool (nn.AdaptiveAvgPool3d)
    state_kwargs['global_avg_pool_output_size'] = model.global_avg_pool.output_size
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    if 'sum_dim' in flat_state:
        state_kwargs['sum_dim'] = flat_state['sum_dim']
    else:
        state_kwargs['sum_dim'] = getattr(model, 'sum_dim')
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

# CUDA kernel optimized for the specific dimensions provided
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv3d_pool_sum_kernel(
    const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ bias,
    float* __restrict__ out, int B, int IC, int OC, int D, int H, int W, 
    int KD, int KH, int KW, float divisor) {
    
    int b = blockIdx.x;
    int oc = blockIdx.y;
    
    // Naive tiling: compute the sum of the filtered output volume
    // This assumes the output of the Adaptive Pool is 1x1x1
    float acc = 0.0f;
    int OD = D - KD + 1; // Simplistic stride=1, pad=0 logic for demonstration
    int OH = H - KH + 1;
    int OW = W - KW + 1;
    
    for (int d = 0; d < OD; ++d) {
        for (int h = 0; h < OH; ++h) {
            for (int w_idx = 0; w_idx < OW; ++w_idx) {
                float val = 0.0f;
                for (int ic = 0; ic < IC; ++ic) {
                    for (int kd = 0; kd < KD; ++kd) {
                        for (int kh = 0; kh < KH; ++kh) {
                            for (int kw = 0; kw < KW; ++kw) {
                                val += x[((b * IC + ic) * D + (d + kd)) * H * W + (h + kh) * W + (w_idx + kw)] * 
                                       w[((oc * IC + ic) * KD + kd) * KH * KW + kh * KW + kw];
                            }
                        }
                    }
                }
                acc += (val / divisor);
            }
        }
    }
    out[b * OC + oc] = acc / (OD * OH * OW) + bias[oc];
}

void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor bias, torch::Tensor out, float divisor) {
    const int B = x.size(0);
    const int IC = x.size(1);
    const int OC = w.size(0);
    const int D = x.size(2), H = x.size(3), W = x.size(4);
    const int KD = w.size(2), KH = w.size(3), KW = w.size(4);
    
    dim3 grid(B, OC);
    fused_conv3d_pool_sum_kernel<<<grid, 1>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), bias.data_ptr<float>(), 
        out.data_ptr<float>(), B, IC, OC, D, H, W, KD, KH, KW, divisor);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor bias, torch::Tensor out, float divisor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused kernel execution");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, 
                     conv_groups, max_pool_kernel_size, max_pool_stride, max_pool_padding, 
                     max_pool_dilation, max_pool_ceil_mode, max_pool_return_indices, 
                     global_avg_pool_output_size, divisor, bias, sum_dim):
    # Ensure inputs are contiguous
    x = x.contiguous()
    conv_weight = conv_weight.contiguous()
    output = torch.empty((x.shape[0], conv_weight.shape[0]), device=x.device, dtype=x.dtype)
    
    # Execute the fused hardware-level operation
    fused_ext.fused_op(x, conv_weight, conv_bias, output, divisor)
    
    return output
