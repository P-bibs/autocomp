# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_144040/code_3.py
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

# Optimized CUDA kernel with tiled shared memory convolution
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 16
#define MAX_KERNEL_SIZE 7

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int batch, int in_c, int in_h, int in_w,
    int out_c, int k, float sub1, float sub2) {

    int out_h_val = in_h - k + 1;
    int out_w_val = in_w - k + 1;
    long total_elements = (long)batch * out_c * out_h_val * out_w_val;
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Shared memory for input tile and weights
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + (TILE_SIZE + MAX_KERNEL_SIZE - 1) * (TILE_SIZE + MAX_KERNEL_SIZE - 1);
    
    // Calculate output position
    int out_idx = bid * blockDim.x + tid;
    if (out_idx >= total_elements) return;

    int tmp = out_idx;
    int ow = tmp % out_w_val; tmp /= out_w_val;
    int oh = tmp % out_h_val; tmp /= out_h_val;
    int oc = tmp % out_c; tmp /= out_c;
    int b = tmp;

    float acc = bias[oc];
    
    // Load weights into shared memory (coalesced access)
    for (int i = tid; i < k * k * in_c; i += blockDim.x) {
        shared_weight[i] = weight[oc * in_c * k * k + i];
    }
    
    // Load input tile into shared memory
    int shared_h = TILE_SIZE + k - 1;
    int shared_w = TILE_SIZE + k - 1;
    
    for (int i = tid; i < shared_h * shared_w; i += blockDim.x) {
        int local_h = i / shared_w;
        int local_w = i % shared_w;
        int global_h = oh + local_h;
        int global_w = ow + local_w;
        
        float val = 0.0f;
        if (global_h < in_h && global_w < in_w) {
            val = input[((b * in_c) * in_h + global_h) * in_w + global_w];
        }
        shared_input[i] = val;
    }
    
    __syncthreads();
    
    // Perform convolution
    for (int ic = 0; ic < in_c; ++ic) {
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                float in_val = shared_input[(i + (TILE_SIZE + k - 1) * (j))];
                float w_val = shared_weight[(ic * k + i) * k + j];
                acc += in_val * w_val;
            }
        }
    }
    
    float val = acc - sub1 - sub2;
    // Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    output[out_idx] = val * tanhf(logf(1.0f + expf(val)));
}

void fused_conv_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                     torch::Tensor output, float sub1, float sub2) {
    int batch = input.size(0);
    int in_c = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_c = weight.size(0);
    int k = weight.size(2);
    int out_h = in_h - k + 1;
    int out_w = in_w - k + 1;
    long total_elements = (long)batch * out_c * out_h * out_w;
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    // Calculate shared memory size
    int shared_input_size = (TILE_SIZE + k - 1) * (TILE_SIZE + k - 1);
    int shared_weight_size = out_c * in_c * k * k;
    size_t shared_mem_size = (shared_input_size + in_c * k * k) * sizeof(float);
    
    fused_conv_mish_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch, in_c, in_h, in_w, out_c, k, sub1, sub2);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_conv_mish(torch::Tensor i, torch::Tensor w, torch::Tensor b, torch::Tensor o, float s1, float s2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_mish", &fused_conv_mish, "Fused Convolution, Subtraction, and Mish with Tiled Shared Memory");
}
"""

fused_ext = load_inline(
    name='fused_ext', 
    cpp_sources=cpp_source, 
    cuda_sources=cuda_source, 
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=0, 
                     conv_dilation=1, conv_groups=1, subtract_value_1, subtract_value_2):
    # This assumes standard 3x3 kernel, stride 1, padding 0 to match the kernel requirement
    batch, _, h, w = x.shape
    k = conv_weight.shape[2]
    out = torch.empty((batch, conv_weight.size(0), h - k + 1, w - k + 1), device=x.device)
    fused_ext.fused_conv_mish(x, conv_weight, conv_bias, out, subtract_value_1, subtract_value_2)
    return out
