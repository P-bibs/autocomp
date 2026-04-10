# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_134740/code_6.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 8
#define THREADS_PER_BLOCK 256

__global__ void fused_conv_transpose3d_post_kernel(
    const float4* __restrict__ input,
    const float4* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ post_bias,
    float4* __restrict__ output,
    int B, int Ci, int Co, 
    int Din, int Hin, int Win,
    int Dout, int Hout, int Wout,
    int KD, int KH, int KW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w
) {
    // Calculate output indices
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs_float4 = (B * Co * Dout * Hout * Wout + 3) / 4;
    
    if (tid >= total_outputs_float4) return;
    
    int elem_idx = tid * 4;
    
    // Decode output position
    int spatial_out = Dout * Hout * Wout;
    int co_base = (elem_idx / spatial_out) % Co;
    int spatial_idx_base = elem_idx % spatial_out;
    int b_base = elem_idx / (Co * spatial_out);
    
    float4 result;
    float* result_vals = reinterpret_cast<float*>(&result);
    
    // Process 4 elements
    for (int i = 0; i < 4; i++) {
        int elem_idx_i = elem_idx + i;
        if (elem_idx_i >= B * Co * Dout * Hout * Wout) {
            result_vals[i] = 0.0f;
            continue;
        }
        
        int spatial_idx = elem_idx_i % spatial_out;
        int w_out = spatial_idx % Wout;
        spatial_idx /= Wout;
        int h_out = spatial_idx % Hout;
        int d_out = spatial_idx / Hout;
        int co = (elem_idx_i / spatial_out) % Co;
        int b = elem_idx_i / (Co * spatial_out);
        
        float val = 0.0f;
        
        // Direct transposed convolution computation
        for (int ci = 0; ci < Ci; ++ci) {
            for (int kd = 0; kd < KD; ++kd) {
                for (int kh = 0; kh < KH; ++kh) {
                    for (int kw = 0; kw < KW; ++kw) {
                        // Map output coordinates to input coordinates
                        int d_in = d_out - kd + pad_d;
                        int h_in = h_out - kh + pad_h;
                        int w_in = w_out - kw + pad_w;
                        
                        // Check if the input position is valid
                        if (d_in >= 0 && d_in < Din && 
                            h_in >= 0 && h_in < Hin && 
                            w_in >= 0 && w_in < Win) {
                            // Calculate input index
                            int input_idx = ((b * Ci + ci) * Din + d_in) * Hin * Win + h_in * Win + w_in;
                            
                            // Calculate weight index (transposed)
                            int weight_idx = ((ci * Co + co) * KD + kd) * KH * KW + kh * KW + kw;
                            
                            val += __ldg(&reinterpret_cast<const float*>(input)[input_idx]) * 
                                   __ldg(&reinterpret_cast<const float*>(weight)[weight_idx]);
                        }
                    }
                }
            }
        }
        
        // Add bias from conv_transpose
        val += conv_bias[co];
        
        // Apply fused post-processing: ((x + b) + x) * x + x
        float post_b = post_bias[co];
        result_vals[i] = ((val + post_b) + val) * val + val;
    }
    
    output[tid] = result;
}

void launch_fused_conv_transpose3d_post(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w
) {
    int B = input.size(0);
    int Ci = input.size(1);
    int Din = input.size(2);
    int Hin = input.size(3);
    int Win = input.size(4);
    
    int Co = weight.size(1); // Note: transposed conv weight shape is (Ci, Co, KD, KH, KW)
    int KD = weight.size(2);
    int KH = weight.size(3);
    int KW = weight.size(4);
    
    int Dout = output.size(2);
    int Hout = output.size(3);
    int Wout = output.size(4);
    
    int total_outputs = B * Co * Dout * Hout * Wout;
    int total_outputs_float4 = (total_outputs + 3) / 4;
    
    int threads = THREADS_PER_BLOCK;
    int blocks = (total_outputs_float4 + threads - 1) / threads;
    
    // Limit blocks for better occupancy on RTX 2080Ti
    blocks = min(blocks, 65535);
    
    const float4* input_ptr = reinterpret_cast<const float4*>(input.data_ptr<float>());
    const float4* weight_ptr = reinterpret_cast<const float4*>(weight.data_ptr<float>());
    float4* output_ptr = reinterpret_cast<float4*>(output.data_ptr<float>());
    
    fused_conv_transpose3d_post_kernel<<<blocks, threads>>>(
        input_ptr,
        weight_ptr,
        conv_bias.data_ptr<float>(),
        post_bias.data_ptr<float>(),
        output_ptr,
        B, Ci, Co,
        Din, Hin, Win,
        Dout, Hout, Wout,
        KD, KH, KW,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_fused_conv_transpose3d_post(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_post", &launch_fused_conv_transpose3d_post, "Fused 3D transposed convolution with post-processing");
}
"""

fused_ext = load_inline(
    name='fused_conv_transpose3d_post_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
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
    # Validate groups (only supporting groups=1 for this implementation)
    if conv_transpose_groups != 1:
        raise ValueError("Only conv_transpose_groups=1 is supported")
    
    # Validate dilation (only supporting dilation=1 for this implementation)
    if conv_transpose_dilation != (1, 1, 1):
        raise ValueError("Only conv_transpose_dilation=(1,1,1) is supported")
    
    # Calculate output dimensions for transposed convolution
    stride_d, stride_h, stride_w = conv_transpose_stride
    pad_d, pad_h, pad_w = conv_transpose_padding
    out_pad_d, out_pad_h, out_pad_w = conv_transpose_output_padding
    
    B, Ci, Din, Hin, Win = x.shape
    Co, _, KD, KH, KW = conv_transpose_weight.shape
    
    Dout = (Din - 1) * stride_d - 2 * pad_d + KD + out_pad_d
    Hout = (Hin - 1) * stride_h - 2 * pad_h + KH + out_pad_h
    Wout = (Win - 1) * stride_w - 2 * pad_w + KW + out_pad_w
    
    # Create output tensor
    output = torch.empty((B, Co, Dout, Hout, Wout), dtype=torch.float32, device='cuda')
    
    # Flatten bias for simplified kernel indexing
    bias_flat = bias.view(-1)
    
    # Launch fused kernel
    fused_ext.fused_conv_transpose3d_post(
        x, conv_transpose_weight, conv_transpose_bias, bias_flat, output,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w
    )
    
    return output

# Model configuration
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
