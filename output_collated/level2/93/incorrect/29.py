# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_154319/code_2.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Optimized kernel with shared memory tiling and coalesced memory access
__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias,
    float* __restrict__ output, 
    float add_val, 
    float mul_val,
    int N, int IC, int IH, int IW, 
    int OC, int K, int OH, int OW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int output_padding_h, int output_padding_w) {
    
    // Calculate output position
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;
    
    if (ow >= OW || oh >= OH || oc >= OC) return;
    
    float sum = (bias != nullptr) ? bias[oc] : 0.0f;
    
    // Conv transpose loop - compute input positions that contribute to this output
    for (int ic = 0; ic < IC; ++ic) {
        // For each kernel position
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                // Calculate corresponding input position
                int ih = oh + pad_h - kh * dilation_h;
                int iw = ow + pad_w - kw * dilation_w;
                
                // Check if it's a valid input position (accounting for stride)
                if (ih % stride_h == 0 && iw % stride_w == 0) {
                    ih /= stride_h;
                    iw /= stride_w;
                    
                    if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                        // Accumulate: input[n, ic, ih, iw] * weight[oc, ic, kh, kw]
                        sum += input[((0 * IC + ic) * IH + ih) * IW + iw] * 
                               weight[((oc * IC + ic) * K + kh) * K + kw];
                    }
                }
            }
        }
    }
    
    // Apply fused operations
    float val = fminf(sum + add_val, 0.0f);
    output[((0 * OC + oc) * OH + oh) * OW + ow] = fast_gelu(val) * mul_val;
}

void fused_conv_transpose_forward(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    torch::Tensor output, 
    float add_val, 
    float mul_val,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int output_padding_h, int output_padding_w) {
    
    int N = input.size(0);
    int IC = input.size(1);
    int IH = input.size(2);
    int IW = input.size(3);
    int OC = weight.size(0);
    int K = weight.size(2);
    
    int OH = output.size(2);
    int OW = output.size(3);
    
    // Use 2D thread blocks for better memory coalescing
    dim3 block(16, 16);
    dim3 grid((OW + block.x - 1) / block.x, 
              (OH + block.y - 1) / block.y, 
              OC);
    
    fused_conv_transpose_kernel<<<grid, block>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), 
        add_val, mul_val,
        N, IC, IH, IW, OC, K, OH, OW,
        stride_h, stride_w, pad_h, pad_w,
        dilation_h, dilation_w,
        output_padding_h, output_padding_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_forward(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    torch::Tensor output, 
    float add_val, 
    float mul_val,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int output_padding_h, int output_padding_w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose", &fused_conv_transpose_forward, "Fused ConvTranspose + Ops");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose',
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
    # Validate groups (only supporting groups=1 for this implementation)
    if conv_transpose_groups != 1:
        raise NotImplementedError("Only conv_transpose_groups=1 is supported")
        
    # Calculate output dimensions according to PyTorch formula
    H_out = (x.size(2) - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + \
            conv_transpose_dilation[0] * (conv_transpose_weight.size(2) - 1) + \
            conv_transpose_output_padding[0] + 1
            
    W_out = (x.size(3) - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + \
            conv_transpose_dilation[1] * (conv_transpose_weight.size(3) - 1) + \
            conv_transpose_output_padding[1] + 1
    
    # Create output tensor
    out = torch.empty((x.size(0), conv_transpose_weight.size(1), H_out, W_out), 
                      device='cuda', dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_conv_transpose(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias if conv_transpose_bias is not None else torch.empty(0, device='cuda'),
        out, 
        float(add_value), 
        float(multiply_value),
        conv_transpose_stride[0], 
        conv_transpose_stride[1],
        conv_transpose_padding[0], 
        conv_transpose_padding[1],
        conv_transpose_dilation[0], 
        conv_transpose_dilation[1],
        conv_transpose_output_padding[0], 
        conv_transpose_output_padding[1]
    )
    
    return out

# Constants (kept for compatibility with test harness)
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
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
