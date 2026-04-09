# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_055948/code_9.py
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

# The CUDA kernel performs a fused operation:
# 1. 3D Convolution (Naive implementation optimized for small kernel sizes)
# 2. Division
# 3. Max pooling
# 4. Global Average Pooling (assuming output 1x1x1 as per standard usage in such ops)
# 5. Bias addition
# 6. Summation over the specified dimension

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_fwd_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias_add,
    float* __restrict__ output,
    int B, int Ci, int Co, int D, int H, int W,
    int Kd, int Kh, int Kw,
    int Sd, int Sh, int Sw,
    int Pd, int Ph, int Pw,
    float divisor,
    int Od, int Oh, int Ow,
    int Pd_pool, int Ph_pool, int Pw_pool,
    int Kd_pool, int Kh_pool, int Kw_pool,
    int Sd_pool, int Sh_pool, int Sw_pool) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * Co * Od * Oh * Ow;
    if (tid >= total_elements) return;

    int tmp = tid;
    int ow = tmp % Ow; tmp /= Ow;
    int oh = tmp % Oh; tmp /= Oh;
    int od = tmp % Od; tmp /= Od;
    int co = tmp % Co; tmp /= Co;
    int b  = tmp;

    // Output of max_pool corresponds to a spatial region in conv output
    // Perform MaxPool logic: find max over the pooling window
    float max_val = -1e38f;

    for (int pd = 0; pd < Kd_pool; ++pd) {
    for (int ph = 0; ph < Kh_pool; ++ph) {
    for (int pw = 0; pw < Kw_pool; ++pw) {
        int conv_d = od * Sd_pool + pd - Pd_pool;
        int conv_h = oh * Sh_pool + ph - Ph_pool;
        int conv_w = ow * Sw_pool + pw - Pw_pool;

        // Perform 3D Conv at this point
        float val = 0.0f;
        for (int kd = 0; kd < Kd; ++kd) {
        for (int kh = 0; kh < Kh; ++kh) {
        for (int kw = 0; kw < Kw; ++kw) {
            int in_d = conv_d * Sd + kd - Pd;
            int in_h = conv_h * Sh + kh - Ph;
            int in_w = conv_w * Sw + kw - Pw;

            if (in_d >= 0 && in_d < D && in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                for (int ci = 0; ci < Ci; ++ci) {
                    float in_pixel = input[b * (Ci * D * H * W) + ci * (D * H * W) + in_d * (H * W) + in_h * W + in_w];
                    float w_pixel = weight[co * (Ci * Kd * Kh * Kw) + ci * (Kd * Kh * Kw) + kd * (Kh * Kw) + kh * Kw + kw];
                    val += in_pixel * w_pixel;
                }
            }
        }}}
        val = (val + conv_bias[co]) / divisor;
        if (val > max_val) max_val = val;
    }}}
    
    // Bias addition and final global avg pool (average of 1x1x1 is itself)
    output[tid] = max_val + bias_add[co];
}
"""

cpp_source = r"""
void fused_fwd_kernel(const float* input, const float* weight, const float* conv_bias, const float* bias_add, float* output, 
                      int B, int Ci, int Co, int D, int H, int W, int Kd, int Kh, int Kw, int Sd, int Sh, int Sw, int Pd, int Ph, int Pw,
                      float divisor, int Od, int Oh, int Ow, int Pd_p, int Ph_p, int Pw_p, int Kd_p, int Kh_p, int Kw_p, int Sd_p, int Sh_p, int Sw_p);

void fused_op(torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias, torch::Tensor bias_add, torch::Tensor output, float divisor,
              int B, int Ci, int Co, int D, int H, int W, int Kd, int Kh, int Kw, int Sd, int Sh, int Sw, int Pd, int Ph, int Pw,
              int Od, int Oh, int Ow, int Pd_p, int Ph_p, int Pw_p, int Kd_p, int Kh_p, int Kw_p, int Sd_p, int Sh_p, int Sw_p) {
    int total = B * Co * Od * Oh * Ow;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fused_fwd_kernel(input.data_ptr<float>(), weight.data_ptr<float>(), conv_bias.data_ptr<float>(), bias_add.data_ptr<float>(), output.data_ptr<float>(),
                     B, Ci, Co, D, H, W, Kd, Kh, Kw, Sd, Sh, Sw, Pd, Ph, Pw, divisor, Od, Oh, Ow, Pd_p, Ph_p, Pw_p, Kd_p, Kh_p, Kw_p, Sd_p, Sh_p, Sw_p);
}
"""

module = load_inline(name='fused_op_lib', cpp_sources=cpp_source, cuda_sources=cuda_source, functions=['fused_op'], with_cuda=True)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups, max_pool_kernel_size, max_pool_stride, max_pool_padding, max_pool_dilation, max_pool_ceil_mode, max_pool_return_indices, global_avg_pool_output_size, divisor, bias, sum_dim):
    B, Ci, D, H, W = x.shape
    Co, _, Kd, Kh, Kw = conv_weight.shape
    
    # Calculate output dims
    Od, Oh, Ow = global_avg_pool_output_size
    output = torch.zeros(B, Co, Od, Oh, Ow, device=x.device, dtype=x.dtype)
    
    module.fused_op(x, conv_weight, conv_bias, bias, output, divisor,
                    B, Ci, Co, D, H, W, Kd, Kh, Kw, conv_stride[0], conv_stride[1], conv_stride[2], conv_padding[0], conv_padding[1], conv_padding[2],
                    Od, Oh, Ow, max_pool_padding[0], max_pool_padding[1], max_pool_padding[2], max_pool_kernel_size[0], max_pool_kernel_size[1], max_pool_kernel_size[2], 
                    max_pool_stride[0], max_pool_stride[1], max_pool_stride[2])
    
    return torch.sum(output, dim=sum_dim)
