# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152911/code_4.py
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

# The fused CUDA kernel handles the spatial computation and the pointwise activations
# to avoid global memory round-trips between these stages.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float add_value,
    const float multiply_value,
    const int N, const int IC, const int OC, 
    const int IH, const int IW,
    const int OH, const int OW,
    const int K, const int stride) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * OC * OH * OW;
    if (idx >= total_elements) return;

    // Decode linear index to 4D tensor coordinates
    int temp = idx;
    int ow = temp % OW; temp /= OW;
    int oh = temp % OH; temp /= OH;
    int oc = temp % OC; temp /= OC;
    int n = temp;

    float val = bias[oc];

    // Transposed Convolution logic:
    // For each output pixel, accumulate contributions from input pixels that influence it.
    for (int ic = 0; ic < IC; ++ic) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int ih = oh - kh;
                int iw = ow - kw;
                
                if (ih % stride == 0 && iw % stride == 0) {
                    int ih_scaled = ih / stride;
                    int iw_scaled = iw / stride;
                    
                    if (ih_scaled >= 0 && ih_scaled < IH && iw_scaled >= 0 && iw_scaled < IW) {
                        int in_idx = ((n * IC + ic) * IH + ih_scaled) * IW + iw_scaled;
                        int wt_idx = ((ic * OC + oc) * K + kh) * K + kw;
                        val += input[in_idx] * weight[wt_idx];
                    }
                }
            }
        }
    }

    // Post-processing fusion
    val += add_value;
    val = fminf(val, 0.0f);
    val = gelu(val);
    output[idx] = val * multiply_value;
}

void launch_fused_conv(
    const torch::Tensor input, const torch::Tensor weight, const torch::Tensor bias,
    torch::Tensor output, float add_value, float multiply_value, int K, int stride) {
    
    int N = input.size(0); int IC = input.size(1);
    int IH = input.size(2); int IW = input.size(3);
    int OC = output.size(1);
    int OH = output.size(2); int OW = output.size(3);
    
    int total_elements = N * OC * OH * OW;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), add_value, multiply_value,
        N, IC, OC, IH, IW, OH, OW, K, stride
    );
}
"""

cpp_source = r"""
void launch_fused_conv(const torch::Tensor input, const torch::Tensor weight, const torch::Tensor bias,
                       torch::Tensor output, float add_value, float multiply_value, int K, int stride);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_conv", &launch_fused_conv, "Fused Transposed Conv + Ops");
}
"""

fused_ext = load_inline(
    name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(
    x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride,
    conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups,
    conv_transpose_dilation, add_value, multiply_value
):
    N, IC, IH, IW = x.shape
    OC, _, K, _ = conv_transpose_weight.shape
    stride = conv_transpose_stride[0]
    # Simple calculation for output spatial size
    OH = (IH - 1) * stride - 2 * conv_transpose_padding[0] + K + conv_transpose_output_padding[0]
    OW = (IW - 1) * stride - 2 * conv_transpose_padding[1] + K + conv_transpose_output_padding[1]
    
    output = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
    fused_ext.launch_fused_conv(x, conv_transpose_weight, conv_transpose_bias, output, add_value, multiply_value, K, stride)
    return output
