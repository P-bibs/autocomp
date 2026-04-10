# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_164831/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape', 'scaling_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias', 'scaling_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

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
    # State for conv_transpose (nn.ConvTranspose2d)
    if 'conv_transpose_weight' in flat_state:
        state_kwargs['conv_transpose_weight'] = flat_state['conv_transpose_weight']
    else:
        state_kwargs['conv_transpose_weight'] = getattr(model.conv_transpose, 'weight', None)
    if 'conv_transpose_bias' in flat_state:
        state_kwargs['conv_transpose_bias'] = flat_state['conv_transpose_bias']
    else:
        state_kwargs['conv_transpose_bias'] = getattr(model.conv_transpose, 'bias', None)
    state_kwargs['conv_transpose_stride'] = model.conv_transpose.stride
    state_kwargs['conv_transpose_padding'] = model.conv_transpose.padding
    state_kwargs['conv_transpose_output_padding'] = model.conv_transpose.output_padding
    state_kwargs['conv_transpose_groups'] = model.conv_transpose.groups
    state_kwargs['conv_transpose_dilation'] = model.conv_transpose.dilation
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    if 'scaling_factor' in flat_state:
        state_kwargs['scaling_factor'] = flat_state['scaling_factor']
    else:
        state_kwargs['scaling_factor'] = getattr(model, 'scaling_factor')
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

# The fused kernel implements a direct 2D transposed convolution with bias, 
# double-clamp post-processing, and scaling fusion.
# Logic: 
# output = clamp(clamp(conv(x) + bias, 0, 1) * s, 0, 1) / s
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fused_conv_tr_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    const float* __restrict__ weight, const float* __restrict__ bias,
    float scaling_factor, int N, int C, int H, int W, int OC, int K, int S, int P) {
    
    int oc = blockIdx.z;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;

    // Output dimension calculation for Transposed Conv
    int OH = (H - 1) * S - 2 * P + K;
    int OW = (W - 1) * S - 2 * P + K;

    if (oh < OH && ow < OW) {
        float val = 0.0f;
        
        // Direct implementation of transposed convolution
        for (int c = 0; c < C; ++c) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int ih = oh + P - kh;
                    int iw = ow + P - kw;
                    
                    if (ih % S == 0 && iw % S == 0) {
                        int i_idx = ih / S;
                        int j_idx = iw / S;
                        if (i_idx >= 0 && i_idx < H && j_idx >= 0 && j_idx < W) {
                            val += input[c * H * W + i_idx * W + j_idx] * 
                                   weight[c * (OC * K * K) + oc * (K * K) + kh * K + kw];
                        }
                    }
                }
            }
        }
        
        val += bias[oc];
        // Fusion: clamp(0, 1) -> scale -> clamp(0, 1) -> unscale
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        val *= scaling_factor;
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        val /= scaling_factor;
        
        output[oc * OH * OW + oh * OW + ow] = val;
    }
}

void fused_op(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
              torch::Tensor output, float scaling_factor, 
              int stride, int padding) {
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int OC = weight.size(1);
    int K = weight.size(2);
    int OH = (H - 1) * stride - 2 * padding + K;
    int OW = (W - 1) * stride - 2 * padding + K;

    dim3 block(16, 16);
    dim3 grid((OW + 15) / 16, (OH + 15) / 16, OC);

    fused_conv_tr_kernel<<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), 
        weight.data_ptr<float>(), bias.data_ptr<float>(), 
        scaling_factor, N, C, H, W, OC, K, stride, padding
    );
}
"""

cpp_source = r"""
void fused_op(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
              torch::Tensor output, float scaling_factor, int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused Transposed Conv and Activation");
}
"""

fused_ext = load_inline(
    name='fused_ct_op', 
    cpp_sources=cpp_source, 
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], 
    with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias, scaling_factor):
    
    # Calculate output dimensions
    N, C_in, H, W = x.shape
    OC, C_out, K, _ = conv_transpose_weight.shape
    OH = (H - 1) * conv_transpose_stride - 2 * conv_transpose_padding + K + conv_transpose_output_padding
    OW = (W - 1) * conv_transpose_stride - 2 * conv_transpose_padding + K + conv_transpose_output_padding
    
    output = torch.empty((N, OC, OH, OW), device=x.device)
    
    fused_ext.fused_op(
        x, conv_transpose_weight, conv_transpose_bias.flatten(), 
        output, scaling_factor, conv_transpose_stride, conv_transpose_padding
    )
    
    return output
