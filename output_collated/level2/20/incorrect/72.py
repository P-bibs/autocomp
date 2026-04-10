# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_140117/code_7.py
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

# Fully fused CUDA kernel implementing conv_transpose3d + elementwise operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 16
#define CHANNEL_TILE 32

__global__ void fused_conv_transpose3d_post_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ post_bias,
    float* __restrict__ output,
    int N, int C_in, int C_out, 
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w
) {
    // Shared memory for weight tiling and post_bias
    __shared__ float shared_weight[CHANNEL_TILE][CHANNEL_TILE];
    __shared__ float shared_post_bias[CHANNEL_TILE];
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_elements = N * C_out * D_out * H_out * W_out;
    
    if (out_idx >= total_out_elements) return;
    
    // Calculate output coordinates
    int n = out_idx / (C_out * D_out * H_out * W_out);
    int temp = out_idx % (C_out * D_out * H_out * W_out);
    int c_out = temp / (D_out * H_out * W_out);
    temp = temp % (D_out * H_out * W_out);
    int d_out = temp / (H_out * W_out);
    temp = temp % (H_out * W_out);
    int h_out = temp / W_out;
    int w_out = temp % W_out;
    
    // Load post_bias into shared memory cooperatively
    if (threadIdx.x < CHANNEL_TILE && c_out + threadIdx.x < C_out) {
        shared_post_bias[threadIdx.x] = post_bias[c_out + threadIdx.x];
    }
    __syncthreads();
    
    float acc = 0.0f;
    
    // Perform transposed convolution calculation
    for (int c_in = 0; c_in < C_in; c_in++) {
        for (int kd = 0; kd < kD; kd++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    // Calculate input coordinates
                    int d_in = d_out - kd + padding_d;
                    int h_in = h_out - kh + padding_h;
                    int w_in = w_out - kw + padding_w;
                    
                    // Check if within input bounds
                    if (d_in % stride_d == 0 && h_in % stride_h == 0 && w_in % stride_w == 0) {
                        d_in /= stride_d;
                        h_in /= stride_h;
                        w_in /= stride_w;
                        
                        if (d_in >= 0 && d_in < D_in && 
                            h_in >= 0 && h_in < H_in && 
                            w_in >= 0 && w_in < W_in) {
                            
                            // Calculate indices
                            int input_idx = ((((n * C_in) + c_in) * D_in + d_in) * H_in + h_in) * W_in + w_in;
                            int weight_idx = ((((c_in * C_out) + c_out) * kD + kd) * kH + kh) * kW + kw;
                            
                            // Accumulate
                            acc += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Apply post-processing: ((x + bias) + x) * x + x
    float bias_val = (c_out < C_out) ? shared_post_bias[c_out % CHANNEL_TILE] : 0.0f;
    float x = acc;
    float result = ((x + bias_val) + x) * x + x;
    
    // Store result
    output[out_idx] = result;
}

void fused_conv_transpose3d_post_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w
) {
    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    
    int C_out = weight.size(1);
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);
    
    // Calculate output dimensions
    int D_out = (D_in - 1) * stride_d - 2 * padding_d + kD;
    int H_out = (H_in - 1) * stride_h - 2 * padding_h + kH;
    int W_out = (W_in - 1) * stride_w - 2 * padding_w + kW;
    
    int total_elements = N * C_out * D_out * H_out * W_out;
    
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    fused_conv_transpose3d_post_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        post_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_post_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w
);

torch::Tensor fused_conv_transpose3d_post(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& post_bias,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t padding_d, int64_t padding_h, int64_t padding_w
) {
    int C_out = weight.size(1);
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);
    
    // Calculate output dimensions
    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    
    int D_out = (D_in - 1) * stride_d - 2 * padding_d + kD;
    int H_out = (H_in - 1) * stride_h - 2 * padding_h + kH;
    int W_out = (W_in - 1) * stride_w - 2 * padding_w + kW;
    
    auto output = torch::empty({N, C_out, D_out, H_out, W_out}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    fused_conv_transpose3d_post_forward(
        input, weight, post_bias, output,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_post", &fused_conv_transpose3d_post, 
          "Fused 3D transposed convolution with post-processing");
}
"""

fused_ext = load_inline(
    name='fused_conv_transpose3d_post',
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
    # Ensure all inputs are contiguous
    x = x.contiguous()
    conv_transpose_weight = conv_transpose_weight.contiguous()
    bias = bias.contiguous()
    
    # Extract stride and padding values (assuming uniform values)
    stride_d = conv_transpose_stride[0] if isinstance(conv_transpose_stride, (tuple, list)) else conv_transpose_stride
    stride_h = conv_transpose_stride[1] if isinstance(conv_transpose_stride, (tuple, list)) else conv_transpose_stride
    stride_w = conv_transpose_stride[2] if isinstance(conv_transpose_stride, (tuple, list)) else conv_transpose_stride
    
    padding_d = conv_transpose_padding[0] if isinstance(conv_transpose_padding, (tuple, list)) else conv_transpose_padding
    padding_h = conv_transpose_padding[1] if isinstance(conv_transpose_padding, (tuple, list)) else conv_transpose_padding
    padding_w = conv_transpose_padding[2] if isinstance(conv_transpose_padding, (tuple, list)) else conv_transpose_padding
    
    # Call fused kernel
    return fused_ext.fused_conv_transpose3d_post(
        x, conv_transpose_weight, bias,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w
    )

# Model parameters
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
