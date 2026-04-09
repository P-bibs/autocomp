# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_034041/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'max_pool1_kernel_size', 'max_pool1_stride', 'max_pool1_padding', 'max_pool1_dilation', 'max_pool1_ceil_mode', 'max_pool1_return_indices', 'max_pool2_kernel_size', 'max_pool2_stride', 'max_pool2_padding', 'max_pool2_dilation', 'max_pool2_ceil_mode', 'max_pool2_return_indices']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

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
    # State for max_pool1 (nn.MaxPool3d)
    state_kwargs['max_pool1_kernel_size'] = model.max_pool1.kernel_size
    state_kwargs['max_pool1_stride'] = model.max_pool1.stride
    state_kwargs['max_pool1_padding'] = model.max_pool1.padding
    state_kwargs['max_pool1_dilation'] = model.max_pool1.dilation
    state_kwargs['max_pool1_ceil_mode'] = model.max_pool1.ceil_mode
    state_kwargs['max_pool1_return_indices'] = model.max_pool1.return_indices
    # State for max_pool2 (nn.MaxPool3d)
    state_kwargs['max_pool2_kernel_size'] = model.max_pool2.kernel_size
    state_kwargs['max_pool2_stride'] = model.max_pool2.stride
    state_kwargs['max_pool2_padding'] = model.max_pool2.padding
    state_kwargs['max_pool2_dilation'] = model.max_pool2.dilation
    state_kwargs['max_pool2_ceil_mode'] = model.max_pool2.ceil_mode
    state_kwargs['max_pool2_return_indices'] = model.max_pool2.return_indices
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused conv_transpose3d + max_pool3d + max_pool3d + sum
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>

#define FLT_MAX __FLT_MAX__

__global__ void fused_conv_transpose3d_max_pool3d_sum_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth, 
    int in_height, 
    int in_width,
    int kernel_size_d,
    int kernel_size_h,
    int kernel_size_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int output_padding_d,
    int output_padding_h,
    int output_padding_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int groups,
    int pool1_kernel_size_d,
    int pool1_kernel_size_h,
    int pool1_kernel_size_w,
    int pool1_stride_d,
    int pool1_stride_h,
    int pool1_stride_w,
    int pool1_padding_d,
    int pool1_padding_h,
    int pool1_padding_w,
    int pool1_dilation_d,
    int pool1_dilation_h,
    int pool1_dilation_w,
    int pool2_kernel_size_d,
    int pool2_kernel_size_h,
    int pool2_kernel_size_w,
    int pool2_stride_d,
    int pool2_stride_h,
    int pool2_stride_w,
    int pool2_padding_d,
    int pool2_padding_h,
    int pool2_padding_w,
    int pool2_dilation_d,
    int pool2_dilation_h,
    int pool2_dilation_w
) {
    // Compute output dimensions after conv transpose
    int out_depth = (in_depth - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_size_d - 1) + output_padding_d + 1;
    int out_height = (in_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_size_h - 1) + output_padding_h + 1;
    int out_width = (in_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_size_w - 1) + output_padding_w + 1;
    
    // Compute intermediate pool1 dimensions
    int pool1_depth = (out_depth + 2 * pool1_padding_d - pool1_dilation_d * (pool1_kernel_size_d - 1) - 1) / pool1_stride_d + 1;
    int pool1_height = (out_height + 2 * pool1_padding_h - pool1_dilation_h * (pool1_kernel_size_h - 1) - 1) / pool1_stride_h + 1;
    int pool1_width = (out_width + 2 * pool1_padding_w - pool1_dilation_w * (pool1_kernel_size_w - 1) - 1) / pool1_stride_w + 1;
    
    // Compute final output dimensions
    int final_depth = (pool1_depth + 2 * pool2_padding_d - pool2_dilation_d * (pool2_kernel_size_d - 1) - 1) / pool2_stride_d + 1;
    int final_height = (pool1_height + 2 * pool2_padding_h - pool2_dilation_h * (pool2_kernel_size_h - 1) - 1) / pool2_stride_h + 1;
    int final_width = (pool1_width + 2 * pool2_padding_w - pool2_dilation_w * (pool2_kernel_size_w - 1) - 1) / pool2_stride_w + 1;
    
    int group_size = out_channels / groups;
    
    // Grid-stride loop over output spatial dimensions
    for (int batch_idx = blockIdx.x; batch_idx < batch_size; batch_idx += gridDim.x) {
        for (int fd = blockIdx.y * blockDim.x + threadIdx.x; fd < final_depth; fd += gridDim.y * blockDim.x) {
            for (int fh = blockIdx.z * blockDim.y + threadIdx.y; fh < final_height; fh += gridDim.z * blockDim.y) {
                for (int fw = threadIdx.z; fw < final_width; fw += blockDim.z) {
                    float sum_val = 0.0f;
                    
                    // Loop over output channels
                    for (int out_c = 0; out_c < out_channels; out_c++) {
                        // Apply second max pooling
                        float max_val2 = -FLT_MAX;
                        
                        for (int kd2 = 0; kd2 < pool2_kernel_size_d; kd2++) {
                            for (int kh2 = 0; kh2 < pool2_kernel_size_h; kh2++) {
                                for (int kw2 = 0; kw2 < pool2_kernel_size_w; kw2++) {
                                    // Calculate position in first pooling output
                                    int p1_d = fd * pool2_stride_d - pool2_padding_d + kd2 * pool2_dilation_d;
                                    int p1_h = fh * pool2_stride_h - pool2_padding_h + kh2 * pool2_dilation_h;
                                    int p1_w = fw * pool2_stride_w - pool2_padding_w + kw2 * pool2_dilation_w;
                                    
                                    if (p1_d >= 0 && p1_d < pool1_depth &&
                                        p1_h >= 0 && p1_h < pool1_height &&
                                        p1_w >= 0 && p1_w < pool1_width) {
                                        
                                        // Apply first max pooling
                                        float max_val1 = -FLT_MAX;
                                        
                                        for (int kd1 = 0; kd1 < pool1_kernel_size_d; kd1++) {
                                            for (int kh1 = 0; kh1 < pool1_kernel_size_h; kh1++) {
                                                for (int kw1 = 0; kw1 < pool1_kernel_size_w; kw1++) {
                                                    // Calculate position in conv transpose output
                                                    int ct_d = p1_d * pool1_stride_d - pool1_padding_d + kd1 * pool1_dilation_d;
                                                    int ct_h = p1_h * pool1_stride_h - pool1_padding_h + kh1 * pool1_dilation_h;
                                                    int ct_w = p1_w * pool1_stride_w - pool1_padding_w + kw1 * pool1_dilation_w;
                                                    
                                                    if (ct_d >= 0 && ct_d < out_depth &&
                                                        ct_h >= 0 && ct_h < out_height &&
                                                        ct_w >= 0 && ct_w < out_width) {
                                                        
                                                        // Apply conv transpose
                                                        float conv_val = 0.0f;
                                                        int group_idx = out_c / group_size;
                                                        
                                                        // Perform convolution
                                                        for (int in_c = group_idx * (in_channels / groups); 
                                                             in_c < (group_idx + 1) * (in_channels / groups); 
                                                             in_c++) {
                                                            for (int kd = 0; kd < kernel_size_d; kd++) {
                                                                for (int kh = 0; kh < kernel_size_h; kh++) {
                                                                    for (int kw = 0; kw < kernel_size_w; kw++) {
                                                                        int in_d = ct_d + padding_d - kd * dilation_d;
                                                                        int in_h = ct_h + padding_h - kh * dilation_h;
                                                                        int in_w = ct_w + padding_w - kw * dilation_w;
                                                                        
                                                                        if (in_d % stride_d == 0 && in_h % stride_h == 0 && in_w % stride_w == 0) {
                                                                            in_d /= stride_d;
                                                                            in_h /= stride_h;
                                                                            in_w /= stride_w;
                                                                            
                                                                            if (in_d >= 0 && in_d < in_depth &&
                                                                                in_h >= 0 && in_h < in_height &&
                                                                                in_w >= 0 && in_w < in_width) {
                                                                                int input_idx = batch_idx * in_channels * in_depth * in_height * in_width +
                                                                                                in_c * in_depth * in_height * in_width +
                                                                                                in_d * in_height * in_width +
                                                                                                in_h * in_width + in_w;
                                                                                
                                                                                int weight_idx = out_c * (in_channels / groups) * kernel_size_d * kernel_size_h * kernel_size_w +
                                                                                                 (in_c - group_idx * (in_channels / groups)) * kernel_size_d * kernel_size_h * kernel_size_w +
                                                                                                 kd * kernel_size_h * kernel_size_w +
                                                                                                 kh * kernel_size_w + kw;
                                                                                
                                                                                conv_val += input[input_idx] * weight[weight_idx];
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                        
                                                        // Add bias
                                                        conv_val += bias[out_c];
                                                        max_val1 = fmaxf(max_val1, conv_val);
                                                    }
                                                }
                                            }
                                        }
                                        
                                        max_val2 = fmaxf(max_val2, max_val1);
                                    }
                                }
                            }
                        }
                        
                        sum_val += max_val2;
                    }
                    
                    int out_idx = batch_idx * final_depth * final_height * final_width +
                                  fd * final_height * final_width +
                                  fh * final_width + fw;
                    output[out_idx] = sum_val;
                }
            }
        }
    }
}

void fused_conv_transpose3d_max_pool3d_sum_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_size_d,
    int kernel_size_h,
    int kernel_size_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int output_padding_d,
    int output_padding_h,
    int output_padding_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int groups,
    int pool1_kernel_size_d,
    int pool1_kernel_size_h,
    int pool1_kernel_size_w,
    int pool1_stride_d,
    int pool1_stride_h,
    int pool1_stride_w,
    int pool1_padding_d,
    int pool1_padding_h,
    int pool1_padding_w,
    int pool1_dilation_d,
    int pool1_dilation_h,
    int pool1_dilation_w,
    int pool2_kernel_size_d,
    int pool2_kernel_size_h,
    int pool2_kernel_size_w,
    int pool2_stride_d,
    int pool2_stride_h,
    int pool2_stride_w,
    int pool2_padding_d,
    int pool2_padding_h,
    int pool2_padding_w,
    int pool2_dilation_d,
    int pool2_dilation_h,
    int pool2_dilation_w
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_channels = weight.size(0);
    
    float* input_ptr = input.data_ptr<float>();
    float* weight_ptr = weight.data_ptr<float>();
    float* bias_ptr = bias.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    dim3 blocks(4, 8, 8);
    dim3 threads(8, 8, 8);
    
    fused_conv_transpose3d_max_pool3d_sum_kernel<<<blocks, threads>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        kernel_size_d, kernel_size_h, kernel_size_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w,
        groups,
        pool1_kernel_size_d, pool1_kernel_size_h, pool1_kernel_size_w,
        pool1_stride_d, pool1_stride_h, pool1_stride_w,
        pool1_padding_d, pool1_padding_h, pool1_padding_w,
        pool1_dilation_d, pool1_dilation_h, pool1_dilation_w,
        pool2_kernel_size_d, pool2_kernel_size_h, pool2_kernel_size_w,
        pool2_stride_d, pool2_stride_h, pool2_stride_w,
        pool2_padding_d, pool2_padding_h, pool2_padding_w,
        pool2_dilation_d, pool2_dilation_h, pool2_dilation_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_max_pool3d_sum_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_size_d,
    int kernel_size_h,
    int kernel_size_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int output_padding_d,
    int output_padding_h,
    int output_padding_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int groups,
    int pool1_kernel_size_d,
    int pool1_kernel_size_h,
    int pool1_kernel_size_w,
    int pool1_stride_d,
    int pool1_stride_h,
    int pool1_stride_w,
    int pool1_padding_d,
    int pool1_padding_h,
    int pool1_padding_w,
    int pool1_dilation_d,
    int pool1_dilation_h,
    int pool1_dilation_w,
    int pool2_kernel_size_d,
    int pool2_kernel_size_h,
    int pool2_kernel_size_w,
    int pool2_stride_d,
    int pool2_stride_h,
    int pool2_stride_w,
    int pool2_padding_d,
    int pool2_padding_h,
    int pool2_padding_w,
    int pool2_dilation_d,
    int pool2_dilation_h,
    int pool2_dilation_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_max_pool3d_sum", &fused_conv_transpose3d_max_pool3d_sum_forward, 
          "Fused conv_transpose3d + max_pool3d + max_pool3d + sum");
}
"""

fused_ext = load_inline(
    name='fused_conv_transpose3d_max_pool3d_sum',
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
    max_pool1_kernel_size,
    max_pool1_stride,
    max_pool1_padding,
    max_pool1_dilation,
    max_pool1_ceil_mode,
    max_pool1_return_indices,
    max_pool2_kernel_size,
    max_pool2_stride,
    max_pool2_padding,
    max_pool2_dilation,
    max_pool2_ceil_mode,
    max_pool2_return_indices,
):
    # Extract parameters
    batch_size = x.size(0)
    in_channels = x.size(1)
    in_depth = x.size(2)
    in_height = x.size(3)
    in_width = x.size(4)
    
    out_channels = conv_transpose_weight.size(0)
    kernel_size_d, kernel_size_h, kernel_size_w = conv_transpose_weight.size()[2:]
    
    stride_d, stride_h, stride_w = conv_transpose_stride
    padding_d, padding_h, padding_w = conv_transpose_padding
    output_padding_d, output_padding_h, output_padding_w = conv_transpose_output_padding
    dilation_d, dilation_h, dilation_w = conv_transpose_dilation
    
    # Compute output dimensions after conv transpose
    out_depth = (in_depth - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_size_d - 1) + output_padding_d + 1
    out_height = (in_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_size_h - 1) + output_padding_h + 1
    out_width = (in_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_size_w - 1) + output_padding_w + 1
    
    # Compute intermediate pool1 dimensions
    pool1_kernel_size_d, pool1_kernel_size_h, pool1_kernel_size_w = max_pool1_kernel_size
    pool1_stride_d, pool1_stride_h, pool1_stride_w = max_pool1_stride
    pool1_padding_d, pool1_padding_h, pool1_padding_w = max_pool1_padding
    pool1_dilation_d, pool1_dilation_h, pool1_dilation_w = max_pool1_dilation
    
    pool1_depth = (out_depth + 2 * pool1_padding_d - pool1_dilation_d * (pool1_kernel_size_d - 1) - 1) // pool1_stride_d + 1
    pool1_height = (out_height + 2 * pool1_padding_h - pool1_dilation_h * (pool1_kernel_size_h - 1) - 1) // pool1_stride_h + 1
    pool1_width = (out_width + 2 * pool1_padding_w - pool1_dilation_w * (pool1_kernel_size_w - 1) - 1) // pool1_stride_w + 1
    
    # Compute final output dimensions
    pool2_kernel_size_d, pool2_kernel_size_h, pool2_kernel_size_w = max_pool2_kernel_size
    pool2_stride_d, pool2_stride_h, pool2_stride_w = max_pool2_stride
    pool2_padding_d, pool2_padding_h, pool2_padding_w = max_pool2_padding
    pool2_dilation_d, pool2_dilation_h, pool2_dilation_w = max_pool2_dilation
    
    final_depth = (pool1_depth + 2 * pool2_padding_d - pool2_dilation_d * (pool2_kernel_size_d - 1) - 1) // pool2_stride_d + 1
    final_height = (pool1_height + 2 * pool2_padding_h - pool2_dilation_h * (pool2_kernel_size_h - 1) - 1) // pool2_stride_h + 1
    final_width = (pool1_width + 2 * pool2_padding_w - pool2_dilation_w * (pool2_kernel_size_w - 1) - 1) // pool2_stride_w + 1
    
    # Allocate output tensor (batch, 1, final_depth, final_height, final_width)
    output = torch.zeros(batch_size, 1, final_depth, final_height, final_width, dtype=x.dtype, device=x.device)
    
    # Call fused kernel
    fused_ext.fused_conv_transpose3d_max_pool3d_sum(
        x, conv_transpose_weight, conv_transpose_bias, output,
        kernel_size_d, kernel_size_h, kernel_size_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w,
        conv_transpose_groups,
        pool1_kernel_size_d, pool1_kernel_size_h, pool1_kernel_size_w,
        pool1_stride_d, pool1_stride_h, pool1_stride_w,
        pool1_padding_d, pool1_padding_h, pool1_padding_w,
        pool1_dilation_d, pool1_dilation_h, pool1_dilation_w,
        pool2_kernel_size_d, pool2_kernel_size_h, pool2_kernel_size_w,
        pool2_stride_d, pool2_stride_h, pool2_stride_w,
        pool2_padding_d, pool2_padding_h, pool2_padding_w,
        pool2_dilation_d, pool2_dilation_h, pool2_dilation_w
    )
    
    return output


batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size = 5
stride = 2
padding = 2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
