# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_045050/code_11.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x', 'add_input']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

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
    # State for conv_transpose (nn.ConvTranspose3d)
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

# ----------------------------------------------------------------------
# Inline CUDA / C++ code for the fused kernel.
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------
// Fused kernel: Transposed 3D Conv + Add + Bias + HardSwish
// ---------------------------------------------------------------
__global__ void fused_conv_transpose3d_add_act_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int stride, const int padding, const int dilation,
    const int K) // kernel size
{
    // Each thread handles one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C_out * D_out * H_out * W_out;
    if (idx >= total) return;

    // Decode linear output index to (n, c_out, d_out, h_out, w_out)
    int w_out = idx % W_out; idx /= W_out;
    int h_out = idx % H_out; idx /= H_out;
    int d_out = idx % D_out; idx /= D_out;
    int c_out = idx % C_out; idx /= C_out;
    int n     = idx;

    // Compute the value for this output element
    float conv_result = 0.0f;

    // Iterate through input points that could contribute to this output point
    // Based on the transposed convolution formula:
    // input[d_in][h_in][w_in] contributes to output[d_out][h_out][w_out]
    // where d_out = d_in * stride - padding + kd * dilation
    // => d_in = (d_out + padding - kd * dilation) / stride
    
    for (int kd = 0; kd < K; ++kd) {
        int numerator_d = d_out + padding - kd * dilation;
        if (numerator_d < 0 || numerator_d % stride != 0) continue;
        int d_in = numerator_d / stride;
        if (d_in >= D_in) continue;
        
        for (int kh = 0; kh < K; ++kh) {
            int numerator_h = h_out + padding - kh * dilation;
            if (numerator_h < 0 || numerator_h % stride != 0) continue;
            int h_in = numerator_h / stride;
            if (h_in >= H_in) continue;
            
            for (int kw = 0; kw < K; ++kw) {
                int numerator_w = w_out + padding - kw * dilation;
                if (numerator_w < 0 || numerator_w % stride != 0) continue;
                int w_in = numerator_w / stride;
                if (w_in >= W_in) continue;

                // Loop over input channels
                for (int c_in = 0; c_in < C_in; ++c_in) {
                    // Input index
                    int in_idx = (((n * C_in + c_in) * D_in + d_in) * H_in + h_in) * W_in + w_in;
                    float in_val = input[in_idx];
                    
                    // Weight index: [C_in, C_out, K, K, K]
                    int w_idx = ((((c_in * C_out + c_out) * K + kd) * K + kh) * K + kw);
                    float w_val = weight[w_idx];
                    
                    conv_result += in_val * w_val;
                }
            }
        }
    }

    // Add per-channel bias
    float final_value = conv_result + conv_bias[c_out];

    // Add residual
    int add_idx = (((n * C_out + c_out) * D_out + d_out) * H_out + h_out) * W_out + w_out;
    final_value += add_input[add_idx];

    // Apply HardSwish: x * clamp(x + 3, 0, 6) / 6
    float hs_input = final_value;
    float clipped = fmaxf(0.0f, fminf(6.0f, hs_input + 3.0f));
    float hardswish_result = hs_input * clipped / 6.0f;

    // Write final result
    output[add_idx] = hardswish_result;
}

// Host wrapper
void fused_conv_transpose3d_add_act(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& add_input,
    torch::Tensor& output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int stride, const int padding, const int dilation,
    const int K)
{
    const int block = 256;
    const int grid  = (N * C_out * D_out * H_out * W_out + block - 1) / block;
    fused_conv_transpose3d_add_act_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride, padding, dilation, K);
    cudaDeviceSynchronize();
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_add_act(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& add_input,
    torch::Tensor& output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int stride, const int padding, const int dilation,
    const int K);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_add_act", &fused_conv_transpose3d_add_act,
          "Fused Transposed 3D Conv + Add + Bias + HardSwish");
}
"""

# Compile the extension with aggressive optimisation flags
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# ----------------------------------------------------------------------
# The functional model that will be imported during evaluation
# ----------------------------------------------------------------------
def functional_model(
    x,
    add_input,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    bias,  # unused in original but part of signature
):
    # ------------------------------------------------------------------
    # Move all tensors to the GPU (if they are not already there)
    # ------------------------------------------------------------------
    x = x.cuda()
    add_input = add_input.cuda()
    conv_transpose_weight = conv_transpose_weight.cuda().contiguous()
    conv_transpose_bias = conv_transpose_bias.cuda().contiguous()

    # ------------------------------------------------------------------
    # Extract shapes and parameters
    # ------------------------------------------------------------------
    N = x.size(0)  # batch size
    C_in = x.size(1)
    D_in = x.size(2)
    H_in = x.size(3)
    W_in = x.size(4)

    stride = conv_transpose_stride[0]  # Assume symmetric
    padding = conv_transpose_padding[0]
    dilation = conv_transpose_dilation[0]
    output_padding = conv_transpose_output_padding
    kernel_size = conv_transpose_weight.size(2)  # Assume cubic kernel

    C_out = conv_transpose_weight.size(1)  # output channels

    # Compute output spatial size using the transposed conv formula
    D_out = (D_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

    # ------------------------------------------------------------------
    # Run fused kernel
    # ------------------------------------------------------------------
    final_out = torch.empty(N, C_out, D_out, H_out, W_out,
                            dtype=torch.float32, device='cuda')

    fused_ext.fused_conv_transpose3d_add_act(
        x, conv_transpose_weight, conv_transpose_bias.squeeze(),
        add_input, final_out,
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride, padding, dilation, kernel_size)

    return final_out
