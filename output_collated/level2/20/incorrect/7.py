# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_125325/code_2.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_post_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t num_elements,
    int64_t spatial_size,
    int64_t out_channels
) {
    // Optimization 8: Use grid-stride loops
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < num_elements; i += stride) {
        int64_t channel_idx = (i / spatial_size) % out_channels;
        float x = input[i];
        float b = bias[channel_idx];
        
        // Operation: ((x + b) + x) * x + x = 2*x^2 + x*b + x
        output[i] = ((x + b) + x) * x + x;
    }
}

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int Ci, int Di, int Hi, int Wi,
    int Co, int Do, int Ho, int Wo,
    int Kd, int Kh, int Kw,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    int DoHoWo = Do * Ho * Wo;
    int HoWo = Ho * Wo;
    
    for (int tid = idx; tid < N * Co * DoHoWo; tid += total_threads) {
        int n = tid / (Co * DoHoWo);
        int c = (tid / DoHoWo) % Co;
        int dhw = tid % DoHoWo;
        int d = dhw / HoWo;
        int h = (dhw / Wo) % Ho;
        int w = dhw % Wo;
        
        float sum = 0.0f;
        
        for (int ci = 0; ci < Ci; ci++) {
            for (int kd = 0; kd < Kd; kd++) {
                for (int kh = 0; kh < Kh; kh++) {
                    for (int kw = 0; kw < Kw; kw++) {
                        int in_d = d * stride_d - padding_d + kd * dilation_d;
                        int in_h = h * stride_h - padding_h + kh * dilation_h;
                        int in_w = w * stride_w - padding_w + kw * dilation_w;
                        
                        if (in_d >= 0 && in_d < Di && 
                            in_h >= 0 && in_h < Hi && 
                            in_w >= 0 && in_w < Wi) {
                            
                            int input_idx = n * (Ci * Di * Hi * Wi) + 
                                          ci * (Di * Hi * Wi) + 
                                          in_d * (Hi * Wi) + 
                                          in_h * Wi + 
                                          in_w;
                            
                            int weight_idx = c * (Ci * Kd * Kh * Kw) + 
                                           ci * (Kd * Kh * Kw) + 
                                           kd * (Kh * Kw) + 
                                           kh * Kw + 
                                           kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        sum += bias[c];
        output[tid] = sum;
    }
}

void fused_post_conv(const torch::Tensor& input, const torch::Tensor& bias, torch::Tensor& output) {
    int64_t num_elements = input.numel();
    int64_t spatial_size = input.size(2) * input.size(3) * input.size(4);
    int64_t out_channels = input.size(1);
    
    // Launch configuration for GPU
    int threads_per_block = 256;
    int blocks = 1024; // Fixed number of blocks to saturate the GPU
    
    fused_post_conv_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        spatial_size,
        out_channels
    );
}

void conv_transpose3d_custom(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    int N = input.size(0);
    int Ci = input.size(1);
    int Di = input.size(2);
    int Hi = input.size(3);
    int Wi = input.size(4);
    
    int Co = weight.size(0);
    int Kd = weight.size(2);
    int Kh = weight.size(3);
    int Kw = weight.size(4);
    
    int Do = (Di - 1) * stride_d - 2 * padding_d + dilation_d * (Kd - 1) + output_padding_d + 1;
    int Ho = (Hi - 1) * stride_h - 2 * padding_h + dilation_h * (Kh - 1) + output_padding_h + 1;
    int Wo = (Wi - 1) * stride_w - 2 * padding_w + dilation_w * (Kw - 1) + output_padding_w + 1;
    
    // Ensure output has the correct shape
    output.resize_({N, Co, Do, Ho, Wo});
    
    int threads_per_block = 256;
    int blocks = min(65535, (N * Co * Do * Ho * Wo + threads_per_block - 1) / threads_per_block);
    
    conv_transpose3d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, Ci, Di, Hi, Wi,
        Co, Do, Ho, Wo,
        Kd, Kh, Kw,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_post_conv(const torch::Tensor& input, const torch::Tensor& bias, torch::Tensor& output);
void conv_transpose3d_custom(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv", &fused_post_conv, "Fused post-conv arithmetic with grid-stride");
    m.def("conv_transpose3d_custom", &conv_transpose3d_custom, "Custom 3D transposed convolution");
}
"""

fused_ext = load_inline(
    name='fused_post_conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, 
                     conv_transpose_groups, conv_transpose_dilation, bias):
    
    # Replace F.conv_transpose3d with our custom CUDA implementation
    x_tmp = torch.empty(0, device=x.device, dtype=x.dtype)
    
    # Handle different stride/padding/output_padding/dilation formats
    if isinstance(conv_transpose_stride, int):
        stride_d = stride_h = stride_w = conv_transpose_stride
    else:
        stride_d, stride_h, stride_w = conv_transpose_stride
    
    if isinstance(conv_transpose_padding, int):
        padding_d = padding_h = padding_w = conv_transpose_padding
    else:
        padding_d, padding_h, padding_w = conv_transpose_padding
        
    if isinstance(conv_transpose_output_padding, int):
        output_padding_d = output_padding_h = output_padding_w = conv_transpose_output_padding
    else:
        output_padding_d, output_padding_h, output_padding_w = conv_transpose_output_padding
        
    if isinstance(conv_transpose_dilation, int):
        dilation_d = dilation_h = dilation_w = conv_transpose_dilation
    else:
        dilation_d, dilation_h, dilation_w = conv_transpose_dilation
    
    # Perform custom 3D transposed convolution
    fused_ext.conv_transpose3d_custom(
        x, conv_transpose_weight, conv_transpose_bias, x_tmp,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w
    )
    
    # Perform fused post-processing
    output = torch.empty_like(x_tmp)
    fused_ext.fused_post_conv(x_tmp, bias.view(-1), output)
    return output

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
