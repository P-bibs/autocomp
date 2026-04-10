# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150347/code_21.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Optimized kernel using shared memory tiling for input and weights
__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int in_c, const int in_h, const int in_w,
    const int out_c, const int k,
    const float sub1, const float sub2)
{
    const int out_h = in_h - k + 1;
    const int out_w = in_w - k + 1;

    // Tile block configuration: 16x16 output pixels
    const int oc = blockIdx.z;
    const int oh_start = blockIdx.y * 16;
    const int ow_start = blockIdx.x * 16;
    
    // Shared memory for weights of one channel
    extern __shared__ float s_weights[];

    // Cooperative weight loading
    const int total_weights = k * k * in_c;
    for (int i = threadIdx.y * 16 + threadIdx.x; i < total_weights; i += 256) {
        s_weights[i] = weight[oc * total_weights + i];
    }
    __syncthreads();

    const int oh = oh_start + threadIdx.y;
    const int ow = ow_start + threadIdx.x;

    if (oh < out_h && ow < out_w) {
        float acc = bias[oc];
        
        // Compute convolution
        #pragma unroll
        for (int i = 0; i < k; ++i) {
            #pragma unroll
            for (int j = 0; j < k; ++j) {
                const int ih = oh + i;
                const int iw = ow + j;
                const int base_in = ih * in_w + iw;
                
                #pragma unroll
                for (int ic = 0; ic < in_c; ++ic) {
                    float inp = input[ic * in_h * in_w + base_in];
                    acc += inp * s_weights[(i * k + j) * in_c + ic];
                }
            }
        }

        const float val = acc - sub1 - sub2;
        // Fast approximation or standard Mish: x * tanh(ln(1 + e^x))
        const float mish = val * tanhf(logf(1.0f + expf(val)));
        output[(oc * out_h + oh) * out_w + ow] = mish;
    }
}

void launch_fused_conv(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                       torch::Tensor output, float sub1, float sub2) {
    const int in_c = input.size(1), in_h = input.size(2), in_w = input.size(3);
    const int out_c = weight.size(0), k = weight.size(2);
    const int out_h = in_h - k + 1, out_w = in_w - k + 1;

    dim3 block(16, 16);
    dim3 grid((out_w + 15) / 16, (out_h + 15) / 16, out_c);
    size_t shm_size = k * k * in_c * sizeof(float);
    
    fused_conv_mish_kernel<<<grid, block, shm_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), in_c, in_h, in_w, out_c, k, sub1, sub2);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_fused_conv(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                       torch::Tensor output, float sub1, float sub2);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &launch_fused_conv, "Fused Conv Mish");
}
"""

module = load_inline(name='fused_conv_mish', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
                     extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x: torch.Tensor, *, conv_weight: torch.Tensor, conv_bias: torch.Tensor, 
                     conv_stride: int = 1, conv_padding: int = 0, conv_dilation: int = 1, 
                     conv_groups: int = 1, subtract_value_1: float, subtract_value_2: float) -> torch.Tensor:
    # Prepare inputs: weight [out_c, k, k, in_c]
    w = conv_weight.permute(0, 2, 3, 1).contiguous()
    batch, in_c, h, w_in = x.shape
    k = conv_weight.shape[2]
    out_h, out_w = h - k + 1, w_in - k + 1
    
    out = torch.empty((batch, conv_weight.size(0), out_h, out_w), device=x.device, dtype=x.dtype)
    
    # Process batch by batch to manage kernel constraints
    for b in range(batch):
        module.fused_conv(x[b].unsqueeze(0), w, conv_bias, out[b].unsqueeze(0), subtract_value_1, subtract_value_2)
        
    return out
