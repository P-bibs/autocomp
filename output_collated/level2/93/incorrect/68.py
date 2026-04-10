# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160727/code_11.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'add_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'add_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

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
    if 'add_value' in flat_state:
        state_kwargs['add_value'] = flat_state['add_value']
    else:
        state_kwargs['add_value'] = getattr(model, 'add_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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

# The custom CUDA kernel performs the Transposed Convolution via a gather-style approach,
# followed immediately by the element-wise post-processing (bias, add, min, gelu, mul).
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_transpose_act_kernel(
    const float* __restrict__ input, const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ output, int N, int C_in, int C_out, int H_in, int W_in, 
    int K, int S, float add_val, float mul_val) {
    
    int H_out = (H_in - 1) * S + K;
    int W_out = (W_in - 1) * S + K;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * H_out * W_out;
    
    if (idx < total_elements) {
        int temp = idx;
        int w_out = temp % W_out; temp /= W_out;
        int h_out = temp % H_out; temp /= H_out;
        int c_out = temp % C_out; temp /= C_out;
        int n = temp;

        float val = bias[c_out];
        
        // Transposed Conv calculation (Gather logic)
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int h_in_idx = h_out - kh;
                    int w_in_idx = w_out - kw;
                    if (h_in_idx >= 0 && w_in_idx >= 0 && h_in_idx % S == 0 && w_in_idx % S == 0) {
                        int h_in = h_in_idx / S;
                        int w_in = w_in_idx / S;
                        if (h_in < H_in && w_in < W_in) {
                            val += input[((n * C_in + c_in) * H_in + h_in) * W_in + w_in] * 
                                   weight[((c_out * C_in + c_in) * K + kh) * K + kw];
                        }
                    }
                }
            }
        }
        
        // Fused post-processing
        val += add_val;
        val = fminf(val, 0.0f);
        val = gelu(val);
        output[idx] = val * mul_val;
    }
}

void fused_op(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
              torch::Tensor out, int S, float add_v, float mul_v) {
    int N = x.size(0), C_in = x.size(1), H_in = x.size(2), W_in = x.size(3);
    int C_out = weight.size(0), K = weight.size(2);
    int H_out = (H_in - 1) * S + K, W_out = (W_in - 1) * S + K;
    int total_threads = N * C_out * H_out * W_out;
    int blocks = (total_threads + 255) / 256;
    fused_conv_transpose_act_kernel<<<blocks, 256>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(),
        N, C_in, C_out, H_in, W_in, K, S, add_v, mul_v);
}
"""

cpp_source = r"""
void fused_op(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out, int S, float add_v, float mul_v);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused Transposed Conv + Activation");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    add_value,
    multiply_value,
):
    # Output spatial dimensions for transposed convolution
    H_out = (x.shape[2] - 1) * conv_transpose_stride + conv_transpose_weight.shape[2]
    W_out = (x.shape[3] - 1) * conv_transpose_stride + conv_transpose_weight.shape[2]
    
    out = torch.empty((x.shape[0], conv_transpose_weight.shape[0], H_out, W_out), device=x.device)
    
    # Execute fused custom kernel
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, out, conv_transpose_stride, add_value, multiply_value)
    
    return out
