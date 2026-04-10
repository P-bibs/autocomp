# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160727/code_3.py
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

# Optimization: Merge low-level operations into a single kernel
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Approximation of GELU: 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Simple implementation of Transposed Conv2d + Activation Fusion
// For performance, we use a naive approach suitable for the provided constraints.
__global__ void fused_op_forward_kernel(
    const float* __restrict__ input, const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ output, int N, int C_in, int C_out, int H_in, int W_in, 
    int H_out, int W_out, int K, int S, int pad, float add_val, float mul_val) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N * C_out * H_out * W_out) {
        int temp = idx;
        int w_out = temp % W_out; temp /= W_out;
        int h_out = temp % H_out; temp /= H_out;
        int c_out = temp % C_out; temp /= C_out;
        int n = temp;

        float val = bias[c_out];
        // Conv transpose loop
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int h_in = h_out - kh + pad;
                    int w_in = w_out - kw + pad;
                    if (h_in >= 0 && w_in >= 0 && h_in < H_in * S && w_in < W_in * S && h_in % S == 0 && w_in % S == 0) {
                        h_in /= S; w_in /= S;
                        if (h_in < H_in && w_in < W_in) {
                            val += input[((n * C_in + c_in) * H_in + h_in) * W_in + w_in] * 
                                   weight[((c_in * C_out + c_out) * K + kh) * K + kw];
                        }
                    }
                }
            }
        }
        // Post-processing steps
        val += add_val;
        val = fminf(val, 0.0f); // torch.min(x, 0)
        val = gelu(val);
        output[idx] = val * mul_val;
    }
}

void fused_op_forward(int blocks, int threads, torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor output, int K, int S, int pad, float add_val, float mul_val) {
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = weight.size(1);
    int H_out = (H_in - 1) * S - 2 * pad + K;
    int W_out = (W_in - 1) * S - 2 * pad + K;
    fused_op_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, C_in, C_out, H_in, W_in, H_out, W_out, K, S, pad, add_val, mul_val);
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function in the .cu file
void fused_op_forward(int blocks, int threads, torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor output, int K, int S, int pad, float add_val, float mul_val);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused ConvTranspose2d + Add + Min + GELU + Mul");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
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
    # Only support basic cases for this optimized version
    assert conv_transpose_groups == 1
    assert conv_transpose_dilation == 1
    assert conv_transpose_output_padding == 0
    assert conv_transpose_weight.shape[2] == conv_transpose_weight.shape[3]  # Square kernel
    
    K = conv_transpose_weight.shape[2]
    S = conv_transpose_stride
    pad = conv_transpose_padding
    
    H_out = (x.shape[2] - 1) * S - 2 * pad + K
    W_out = (x.shape[3] - 1) * S - 2 * pad + K
    
    out = torch.empty((x.shape[0], conv_transpose_weight.shape[1], H_out, W_out), device=x.device)
    
    total_threads = out.numel()
    threads_per_block = 256
    blocks = (total_threads + threads_per_block - 1) // threads_per_block
    
    fused_ext.fused_op(blocks, threads_per_block, x, conv_transpose_weight, conv_transpose_bias, 
                       out, K, S, pad, add_value, multiply_value)
    return out

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
