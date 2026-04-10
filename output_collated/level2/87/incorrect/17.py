# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_142828/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'subtract_value_1', 'subtract_value_2']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'subtract_value_1', 'subtract_value_2']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, subtracts two values, applies Mish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

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
    # State for conv (nn.Conv2d)
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
    if 'subtract_value_1' in flat_state:
        state_kwargs['subtract_value_1'] = flat_state['subtract_value_1']
    else:
        state_kwargs['subtract_value_1'] = getattr(model, 'subtract_value_1')
    if 'subtract_value_2' in flat_state:
        state_kwargs['subtract_value_2'] = flat_state['subtract_value_2']
    else:
        state_kwargs['subtract_value_2'] = getattr(model, 'subtract_value_2')
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

# CUDA kernel code
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float mish_impl(float x) {
    float softplus = x > 20.0f ? x : (x < -20.0f ? 0.0f : log1pf(expf(x)));
    return x * tanhf(softplus);
}

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int OC, int KH, int KW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    float subtract_value_1, float subtract_value_2
) {
    int oh = blockIdx.x;
    int ow = blockIdx.y;
    int oc = blockIdx.z;
    
    if (oh >= H || ow >= W || oc >= OC) return;
    
    int thread_id = threadIdx.x;
    float acc = 0.0f;
    
    for (int ic = thread_id; ic < C; ic += blockDim.x) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                int ih = oh * stride_h - pad_h + kh * dilation_h;
                int iw = ow * stride_w - pad_w + kw * dilation_w;
                
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float val = input[((0 * C + ic) * H + ih) * W + iw];
                    float wgt = weight[((oc * C + ic) * KH + kh) * KW + kw];
                    acc += val * wgt;
                }
            }
        }
    }
    
    __shared__ float sdata[256];
    sdata[thread_id] = acc;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_id < s) {
            sdata[thread_id] += sdata[thread_id + s];
        }
        __syncthreads();
    }
    
    if (thread_id == 0) {
        float result = sdata[0] + bias[oc];
        result -= subtract_value_1;
        result -= subtract_value_2;
        result = mish_impl(result);
        output[((0 * OC + oc) * H + oh) * W + ow] = result;
    }
}

void launch_fused_conv_mish(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    float subtract_value_1,
    float subtract_value_2
) {
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int OC = weight.size(0);
    int KH = weight.size(2);
    int KW = weight.size(3);
    
    int OH = (H + 2 * pad_h - dilation_h * (KH - 1) - 1) / stride_h + 1;
    int OW = (W + 2 * pad_w - dilation_w * (KW - 1) - 1) / stride_w + 1;
    
    dim3 grid(OH, OW, OC);
    dim3 block(min(256, C));
    
    for (int n = 0; n < N; n++) {
        auto input_n = input[n];
        auto output_n = output[n];
        
        fused_conv_mish_kernel<<<grid, block>>>(
            input_n.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            output_n.data_ptr<float>(),
            1, C, H, W, OC, KH, KW,
            stride_h, stride_w, pad_h, pad_w,
            dilation_h, dilation_w,
            subtract_value_1, subtract_value_2
        );
    }
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void launch_fused_conv_mish(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    float subtract_value_1,
    float subtract_value_2
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_mish", &launch_fused_conv_mish, "Fused Conv + Mish");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_mish',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    subtract_value_1,
    subtract_value_2,
):
    # Ensure inputs are contiguous
    x = x.contiguous()
    conv_weight = conv_weight.contiguous()
    conv_bias = conv_bias.contiguous()
    
    # Handle stride, padding, dilation tuples
    if isinstance(conv_stride, int):
        stride_h = stride_w = conv_stride
    else:
        stride_h, stride_w = conv_stride
        
    if isinstance(conv_padding, int):
        pad_h = pad_w = conv_padding
    else:
        pad_h, pad_w = conv_padding
        
    if isinstance(conv_dilation, int):
        dilation_h = dilation_w = conv_dilation
    else:
        dilation_h, dilation_w = conv_dilation
    
    # Calculate output dimensions
    N, C, H, W = x.shape
    OC, _, KH, KW = conv_weight.shape
    OH = (H + 2 * pad_h - dilation_h * (KH - 1) - 1) // stride_h + 1
    OW = (W + 2 * pad_w - dilation_w * (KW - 1) - 1) // stride_w + 1
    
    # Create output tensor
    output = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
    
    # Launch fused kernel
    fused_ext.fused_conv_mish(
        x, conv_weight, conv_bias, output,
        stride_h, stride_w, pad_h, pad_w,
        dilation_h, dilation_w,
        subtract_value_1, subtract_value_2
    )
    
    return output

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
