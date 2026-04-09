# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_090933/code_15.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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
# CUDA Kernel: Fused Conv2d, Min Reduction, and Double Tanh
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const bool has_bias,
    const int stride,
    const int padding,
    const int dilation,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int out_height,
    const int out_width,
    float* __restrict__ output)
{
    // Flattened grid: each block corresponds to one spatial pixel (b, oh, ow)
    int b = blockIdx.x / (out_height * out_width);
    int rem = blockIdx.x % (out_height * out_width);
    int oh = rem / out_width;
    int ow = rem % out_width;

    int oc = threadIdx.x; // Each thread computes a specific output channel

    // Convolution: calculate for the assigned output channel
    float sum = has_bias ? bias[oc] : 0.0f;
    int ih_start = oh * stride - padding;
    int iw_start = ow * stride - padding;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            int ih = ih_start + kh * dilation;
            if (ih < 0 || ih >= in_height) continue;
            for (int kw = 0; kw < kernel_size; ++kw) {
                int iw = iw_start + kw * dilation;
                if (iw < 0 || iw >= in_width) continue;
                
                float inp = input[((b * in_channels + ic) * in_height + ih) * in_width + iw];
                float w = weight[((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw];
                sum += inp * w;
            }
        }
    }

    // Shared memory for reduction
    extern __shared__ float s_data[];
    s_data[oc] = sum;
    __syncthreads();

    // Parallel reduction for min
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (oc < s) {
            s_data[oc] = fminf(s_data[oc], s_data[oc + s]);
        }
        __syncthreads();
    }

    // Only thread 0 in the block performs tanh and writes output
    if (oc == 0) {
        float val = s_data[0];
        val = tanhf(val);
        val = tanhf(val);
        output[((b * out_height) + oh) * out_width + ow] = val;
    }
}

void launch_fused_conv(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride, int padding, int dilation, torch::Tensor output)
{
    int batch_size = input.size(0);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int out_h = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_w = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    dim3 grid(batch_size * out_h * out_w);
    dim3 block(out_channels);
    size_t smem = out_channels * sizeof(float);

    fused_conv_min_tanh_kernel<<<grid, block, smem>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        (bias.numel() > 0) ? bias.data_ptr<float>() : nullptr,
        bias.numel() > 0, stride, padding, dilation,
        batch_size, input.size(1), out_channels, in_height, in_width,
        kernel_size, out_h, out_w, output.data_ptr<float>()
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_fused_conv(torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, torch::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &launch_fused_conv, "Fused Conv/Min/Tanh");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    # Ensure inputs are CUDA/contiguous
    x, conv_weight = x.cuda().contiguous(), conv_weight.cuda().contiguous()
    conv_bias = conv_bias.cuda().contiguous() if conv_bias is not None else torch.tensor([], device='cuda')
    
    out_h = (x.size(2) + 2 * conv_padding - conv_dilation * (conv_weight.size(2) - 1) - 1) // conv_stride + 1
    out_w = (x.size(3) + 2 * conv_padding - conv_dilation * (conv_weight.size(3) - 1) - 1) // conv_stride + 1
    output = torch.empty((x.size(0), 1, out_h, out_w), device='cuda')
    
    fused_ext.fused_conv(x, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, output)
    return output

batch_size, in_channels, out_channels, height, width, kernel_size = 128, 16, 64, 256, 256, 3
def get_init_inputs(): return [in_channels, out_channels, kernel_size]
def get_inputs(): return [torch.rand(batch_size, in_channels, height, width)]
