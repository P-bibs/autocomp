# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150347/code_0.py
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

# -------------------------------------------------------------------------
# CUDA kernel: Optimized for memory coalescing and improved performance
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE 16

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int in_c, int in_h, int in_w,
    int out_c, int k, int out_h, int out_w,
    float sub1, float sub2)
{
    // Shared memory to cache kernel weights for the current channel
    extern __shared__ float weight_smem[];
    int weight_vol = k * k * in_c;
    int oc = blockIdx.y;
    int b = blockIdx.z;

    // Coalesced loading of weights into shared memory
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y;
    
    for (int i = tid; i < weight_vol; i += block_size) {
        weight_smem[i] = weight[oc * weight_vol + i];
    }
    __syncthreads();

    // Grid-stride loop with coalesced access pattern
    int total_elements = out_h * out_w;
    int global_tid = (blockIdx.z * gridDim.y + blockIdx.y) * (gridDim.x * block_size) + 
                     blockIdx.x * block_size + tid;
    int grid_stride = gridDim.x * gridDim.y * gridDim.z * block_size;

    for (int idx = global_tid; idx < total_elements; idx += grid_stride) {
        // Coalesced indexing: consecutive threads process consecutive output elements
        int ow = idx % out_w;
        int oh = (idx / out_w) % out_h;
        
        float acc = bias[oc];
        
        // Convolution loop with optimized memory access
        for (int ic = 0; ic < in_c; ++ic) {
            for (int i = 0; i < k; ++i) {
                int ih = oh + i;
                if (ih < 0 || ih >= in_h) continue;
                
                for (int j = 0; j < k; ++j) {
                    int iw = ow + j;
                    if (iw < 0 || iw >= in_w) continue;
                    
                    // Coalesced memory access pattern
                    int input_idx = ((b * in_c + ic) * in_h + ih) * in_w + iw;
                    int weight_idx = ((i * k + j) * in_c + ic);
                    
                    acc += input[input_idx] * weight_smem[weight_idx];
                }
            }
        }

        // Mish activation
        float val = acc - sub1 - sub2;
        int output_idx = ((b * out_c + oc) * out_h + oh) * out_w + ow;
        output[output_idx] = val * tanhf(logf(1.0f + expf(val)));
    }
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

    dim3 threads(TILE, TILE);
    // Optimized grid size for better occupancy and coalescing
    int blocks_x = (out_h * out_w + threads.x * threads.y - 1) / (threads.x * threads.y);
    blocks_x = min(blocks_x, 65535); // Max grid dimension
    dim3 blocks(blocks_x, out_c, batch);
    
    size_t shared_mem_size = k * k * in_c * sizeof(float);
    
    fused_conv_mish_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch, in_c, in_h, in_w, out_c, k, out_h, out_w, sub1, sub2
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_conv_mish(torch::Tensor i, torch::Tensor w, torch::Tensor b, torch::Tensor o, float s1, float s2);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &fused_conv_mish, "Fused Convolution + Mish");
}
"""

fused_ext = load_inline(
    name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_source, 
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=0, 
                     conv_dilation=1, conv_groups=1, subtract_value_1, subtract_value_2):
    # Prepare weights: [out_c, k, k, in_c]
    w_reordered = conv_weight.permute(0, 2, 3, 1).contiguous()
    
    batch, _, h, w = x.shape
    k = conv_weight.shape[2]
    out = torch.empty((batch, conv_weight.size(0), h - k + 1, w - k + 1), device=x.device)
    
    fused_ext.fused_conv(x, w_reordered, conv_bias, out, float(subtract_value_1), float(subtract_value_2))
    return out
