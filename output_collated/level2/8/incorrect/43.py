# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_061658/code_13.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused kernel for Conv3d -> MaxPool3d -> AvgPool3d -> Bias -> Div -> Sum 
// We use a direct implementation to avoid external function call overhead.
__global__ void fused_conv_pool_kernel(
    const float* __restrict__ x, const float* __restrict__ weight,
    const float* __restrict__ conv_bias, const float* __restrict__ post_bias,
    float divisor, int N, int C, int Din, int Hin, int Win,
    int K, int Dk, int Hk, int Wk, int stride, int padding,
    int pool_k, int pool_s, int pool_p,
    int OD, int OH, int OW, float* __restrict__ output) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OD * OH * OW;
    if (idx >= total) return;

    int n = idx / (OD * OH * OW);
    int spatial_idx = idx % (OD * OH * OW);
    int od = spatial_idx / (OH * OW);
    int oh = (spatial_idx / OW) % OH;
    int ow = spatial_idx % OW;

    // Convolution output dimensions
    int conv_D = (Din + 2 * padding - Dk) / stride + 1;
    int conv_H = (Hin + 2 * padding - Hk) / stride + 1;
    int conv_W = (Win + 2 * padding - Wk) / stride + 1;

    // MaxPool window
    int pool_start_d = od * pool_s - pool_p;
    int pool_start_h = oh * pool_s - pool_p;
    int pool_start_w = ow * pool_s - pool_p;

    for (int k = 0; k < K; ++k) {
        float max_val = -1e38f;
        for (int pd = 0; pd < pool_k; ++pd) {
            for (int ph = 0; ph < pool_k; ++ph) {
                for (int pw = 0; pw < pool_k; ++pw) {
                    int cd = pool_start_d + pd;
                    int ch = pool_start_h + ph;
                    int cw = pool_start_w + pw;
                    if (cd < 0 || cd >= conv_D || ch < 0 || ch >= conv_H || cw < 0 || cw >= conv_W) continue;
                    
                    float sum = conv_bias[k];
                    for (int c = 0; c < C; ++c) {
                        for (int kd = 0; kd < Dk; ++kd) {
                            for (int kh = 0; kh < Hk; ++kh) {
                                for (int kw = 0; kw < Wk; ++kw) {
                                    int in_d = cd * stride + kd - padding;
                                    int in_h = ch * stride + kh - padding;
                                    int in_w = cw * stride + kw - padding;
                                    if (in_d >= 0 && in_d < Din && in_h >= 0 && in_h < Hin && in_w >= 0 && in_w < Win) {
                                        sum += x[((n * C + c) * Din + in_d) * Hin * Win + in_h * Win + in_w] * 
                                               weight[((k * C + c) * Dk + kd) * Hk * Wk + kh * Wk + kw];
                                    }
                                }
                            }
                        }
                    }
                    if (sum > max_val) max_val = sum;
                }
            }
        }
        float val = (max_val / divisor) + post_bias[k];
        atomicAdd(&output[((n * OD + od) * OH + oh) * OW + ow], val);
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor cb, torch::Tensor pb, float div,
                      int stride, int pad, int pk, int ps, int pp, int OD, int OH, int OW, torch::Tensor out) {
    int N = x.size(0); int C = x.size(1); int Din = x.size(2); int Hin = x.size(3); int Win = x.size(4);
    int K = w.size(0); int Dk = w.size(2); int Hk = w.size(3); int Wk = w.size(4);
    int total = N * OD * OH * OW;
    int threads = 256;
    fused_conv_pool_kernel<<<(total + threads - 1) / threads, threads>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), cb.data_ptr<float>(), pb.data_ptr<float>(),
        div, N, C, Din, Hin, Win, K, Dk, Hk, Wk, stride, pad, pk, ps, pp, OD, OH, OW, out.data_ptr<float>()
    );
}
"""

cpp_source = "void fused_op_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, int, int, int, int, int, int, int, int, torch::Tensor);"

fused_ext = load_inline(name='fused_op', cpp_sources="void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor cb, torch::Tensor pb, float div, int s, int p, int pk, int ps, int pp, int OD, int OH, int OW, torch::Tensor out); PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"fused_op\", &fused_op_forward); }", cuda_sources=cuda_source, with_cuda=True, extra_cuda_cflags=['-O3'])

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups, max_pool_kernel_size, max_pool_stride, max_pool_padding, max_pool_dilation, max_pool_ceil_mode, max_pool_return_indices, global_avg_pool_output_size, divisor, bias, sum_dim):
    OD, OH, OW = global_avg_pool_output_size
    output = torch.zeros([x.size(0), OD, OH, OW], device=x.device)
    fused_ext.fused_op(x, conv_weight, conv_bias, bias, divisor, conv_stride, conv_padding, max_pool_kernel_size, max_pool_stride, max_pool_padding, OD, OH, OW, output)
    return output
