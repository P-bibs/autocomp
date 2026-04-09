# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_134740/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by a sum, 
    a residual add, a multiplication, and another residual add.
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

# -------------------------------------------------------------------------
# Optimized CUDA kernel – fuse transposed convolution and activation
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ void fused_conv_transpose3d_activation_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ post_bias,
    float* __restrict__ output,
    const int N, const int C_in, const int D_in, const int H_in, const int W_in,
    const int C_out, const int D_out, const int H_out, const int W_out,
    const int K_D, const int K_H, const int K_W,
    const int stride_d, const int stride_h, const int stride_w,
    const int padding_d, const int padding_h, const int padding_w,
    const int output_padding_d, const int output_padding_h, const int output_padding_w,
    const int dilation_d, const int dilation_h, const int dilation_w
) {
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * D_out * H_out * W_out) return;
    
    // Decode 5D output index
    int w_out_idx = idx % W_out;
    int h_out_idx = (idx / W_out) % H_out;
    int d_out_idx = (idx / (W_out * H_out)) % D_out;
    int c_out_idx = (idx / (W_out * H_out * D_out)) % C_out;
    int n_idx = idx / (W_out * H_out * D_out * C_out);

    float sum = 0.0f;
    
    // Iterate over input channels (assuming groups=1)
    for (int c_in = 0; c_in < C_in; ++c_in) {
        // Iterate over kernel dimensions
        for (int kd = 0; kd < K_D; ++kd) {
            for (int kh = 0; kh < K_H; ++kh) {
                for (int kw = 0; kw < K_W; ++kw) {
                    // Map output spatial coordinates to input coordinates
                    // For transposed convolution: 
                    // input_coord = (output_coord + padding - kernel_pos * dilation) / stride
                    int d_in_coord = d_out_idx + padding_d - kd * dilation_d;
                    int h_in_coord = h_out_idx + padding_h - kh * dilation_h;
                    int w_in_coord = w_out_idx + padding_w - kw * dilation_w;
                    
                    // Check if the input coordinate is valid after division by stride
                    if (d_in_coord >= 0 && d_in_coord < D_in * stride_d &&
                        h_in_coord >= 0 && h_in_coord < H_in * stride_h &&
                        w_in_coord >= 0 && w_in_coord < W_in * stride_w &&
                        d_in_coord % stride_d == 0 &&
                        h_in_coord % stride_h == 0 &&
                        w_in_coord % stride_w == 0) {
                        
                        int d_in_final = d_in_coord / stride_d;
                        int h_in_final = h_in_coord / stride_h;
                        int w_in_final = w_in_coord / stride_w;

                        // Bounds check on input tensor
                        if (d_in_final < D_in && h_in_final < H_in && w_in_final < W_in) {
                            // Get input index
                            int input_idx = ((n_idx * C_in + c_in) * D_in + d_in_final) * H_in * W_in + h_in_final * W_in + w_in_final;
                            // Get weight index (flipped for transposed conv)
                            int weight_idx = ((c_in * C_out + c_out_idx) * K_D + kd) * K_H * K_W + kh * K_W + kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }

    // Add bias from convolution
    sum += conv_bias[c_out_idx];
    
    // Apply activation function: ((x + b) + x) * x + x = 2*x*x + b*x + x
    float bias_val = post_bias[c_out_idx];
    float tmp  = sum + bias_val;          // x + b
    float tmp2 = tmp + sum;               // (x + b) + x = 2*x + b
    float res  = tmp2 * sum + sum;        // ((2*x + b) * x) + x

    output[idx] = res;
}

void fused_conv_transpose3d_activation_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& output_padding,
    const std::vector<int64_t>& dilation
) {
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    const int C_out = weight.size(0); // For transposed conv, weight is [in_channels, out_channels, ...]
    const int K_D = weight.size(2);
    const int K_H = weight.size(3);
    const int K_W = weight.size(4);
    
    const int D_out = output.size(2);
    const int H_out = output.size(3);
    const int W_out = output.size(4);
    
    const int num_elements = N * C_out * D_out * H_out * W_out;
    const int threads_per_block = 256;
    const int blocks = CEIL_DIV(num_elements, threads_per_block);
    
    auto stream = c10::cuda::getCurrentCUDAStream();
    fused_conv_transpose3d_activation_kernel<<<blocks, threads_per_block, 0, stream>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        post_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        K_D, K_H, K_W,
        static_cast<int>(stride[0]), static_cast<int>(stride[1]), static_cast<int>(stride[2]),
        static_cast<int>(padding[0]), static_cast<int>(padding[1]), static_cast<int>(padding[2]),
        static_cast<int>(output_padding[0]), static_cast<int>(output_padding[1]), static_cast<int>(output_padding[2]),
        static_cast<int>(dilation[0]), static_cast<int>(dilation[1]), static_cast<int>(dilation[2])
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_activation_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& output_padding,
    const std::vector<int64_t>& dilation);

torch::Tensor fused_conv_transpose3d_activation(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& output_padding,
    const std::vector<int64_t>& dilation) {
    
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(conv_bias.is_contiguous(), "conv_bias must be contiguous");
    TORCH_CHECK(post_bias.is_contiguous(), "post_bias must be contiguous");
    
    // Calculate output dimensions for transposed convolution
    int64_t D_out = (input.size(2) - 1) * stride[0] - 2 * padding[0] + dilation[0] * (weight.size(2) - 1) + output_padding[0] + 1;
    int64_t H_out = (input.size(3) - 1) * stride[1] - 2 * padding[1] + dilation[1] * (weight.size(3) - 1) + output_padding[1] + 1;
    int64_t W_out = (input.size(4) - 1) * stride[2] - 2 * padding[2] + dilation[2] * (weight.size(4) - 1) + output_padding[2] + 1;
    
    auto output = torch::empty({input.size(0), weight.size(0), D_out, H_out, W_out}, input.options());
    
    fused_conv_transpose3d_activation_forward(input, weight, conv_bias, post_bias, output,
                                              stride, padding, output_padding, dilation);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_activation", &fused_conv_transpose3d_activation, "Fused 3D Transposed Conv + Activation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose3d_activation',
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
    bias,
):
    # Flatten bias to a 1-D tensor (required by the kernel)
    bias_flat = bias.view(-1)
    
    # Convert parameters to lists if they are integers
    def to_list(param, ndim=3):
        if isinstance(param, int):
            return [param] * ndim
        return list(param)
    
    conv_transpose_stride = to_list(conv_transpose_stride, 3)
    conv_transpose_padding = to_list(conv_transpose_padding, 3)
    conv_transpose_output_padding = to_list(conv_transpose_output_padding, 3)
    conv_transpose_dilation = to_list(conv_transpose_dilation, 3)
        
    # Call fused kernel that does both transposed convolution and activation
    return fused_ext.fused_conv_transpose3d_activation(
        x.contiguous(),
        conv_transpose_weight.contiguous(),
        conv_transpose_bias.contiguous(),
        bias_flat.contiguous(),
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_dilation
    )

# -------------------------------------------------------------------------
# Helper code (shape parameters, input factories)
# -------------------------------------------------------------------------
batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
