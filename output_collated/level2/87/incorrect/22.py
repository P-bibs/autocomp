# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_142828/code_4.py
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

# ----------------------------------------------------------------------
# Optimized CUDA kernel: Tiled convolution with coalesced memory access
# ----------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_DIM 16
#define HALO (KERN_SIZE / 2)

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch, const int in_c, const int in_h, const int in_w,
    const int out_c, const int k,
    const int out_h, const int out_w,
    const float sub1, const float sub2) {

    // Use constant memory or template param for kernel size
    const int KERN_SIZE = k;
    const int HALO_SIZE = KERN_SIZE / 2;

    // Shared memory for input tile (with halo)
    extern __shared__ float s_input[];
    
    int oc = blockIdx.z; // output channel
    int b = blockIdx.y;  // batch
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;

    // Each thread block handles one output channel and one batch element
    // Grid is (ceil(out_w / TILE_DIM), batch, out_c)

    // Load bias once
    float b_val = bias[oc];

    // Tile coordinates
    int ow_base = blockIdx.x * TILE_DIM;
    int oh_base = 0;

    for (int oh_tile_start = 0; oh_tile_start < out_h; oh_tile_start += TILE_DIM) {
        oh_base = oh_tile_start;
        
        // Collaborative loading of input tile with halo into shared memory
        // Each thread loads multiple elements if needed
        for (int load_idx = tid; load_idx < (TILE_DIM + 2*HALO_SIZE) * (TILE_DIM + 2*HALO_SIZE) * in_c; load_idx += total_threads) {
            int local_c = load_idx % in_c;
            int temp = load_idx / in_c;
            int local_h = temp % (TILE_DIM + 2*HALO_SIZE);
            int local_w = temp / (TILE_DIM + 2*HALO_SIZE);

            int global_h = oh_base + local_h - HALO_SIZE;
            int global_w = ow_base + local_w - HALO_SIZE;

            // Boundary checks
            if (global_h >= 0 && global_h < in_h && global_w >= 0 && global_w < in_w) {
                s_input[load_idx] = input[((b * in_c + local_c) * in_h + global_h) * in_w + global_w];
            } else {
                s_input[load_idx] = 0.0f;
            }
        }
        __syncthreads();

        // Process pixels within the tile
        for (int idx = tid; idx < TILE_DIM * TILE_DIM; idx += total_threads) {
            int tx = idx % TILE_DIM;
            int ty = idx / TILE_DIM;
            int ow = ow_base + tx;
            int oh = oh_base + ty;

            if (ow < out_w && oh < out_h) {
                float acc = b_val;

                // Convolve with weights
                for (int ic = 0; ic < in_c; ++ic) {
                    for (int ky = 0; ky < KERN_SIZE; ++ky) {
                        for (int kx = 0; kx < KERN_SIZE; ++kx) {
                            // Access from shared memory
                            int sh_h = ty + ky;
                            int sh_w = tx + kx;
                            int sh_idx = ((ic * (TILE_DIM + 2*HALO_SIZE) + sh_h) * (TILE_DIM + 2*HALO_SIZE) + sh_w);
                            float in_val = s_input[sh_idx];
                            float w_val = weight[(((oc * in_c) + ic) * KERN_SIZE + ky) * KERN_SIZE + kx];
                            acc += in_val * w_val;
                        }
                    }
                }

                // Fused Mish activation
                float val = acc - sub1 - sub2;
                float softplus_val = logf(1.0f + expf(val));
                output[(((b * out_c + oc) * out_h + oh) * out_w + ow)] = val * tanhf(softplus_val);
            }
        }
        __syncthreads();
    }
}

void fused_conv_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                     torch::Tensor output, float sub1, float sub2) {
    const int batch = input.size(0);
    const int in_c = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_c = weight.size(0);
    const int k = weight.size(2);
    const int out_h = in_h - k + 1;
    const int out_w = in_w - k + 1;

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 grid((out_w + TILE_DIM - 1) / TILE_DIM, batch, out_c);
    
    size_t shared_size = (TILE_DIM + 2*(k/2)) * (TILE_DIM + 2*(k/2)) * in_c * sizeof(float);

    fused_conv_mish_kernel<<<grid, threads, shared_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch, in_c, in_h, in_w, out_c, k, 
        out_h, out_w, sub1, sub2);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_conv_mish(torch::Tensor i, torch::Tensor w, torch::Tensor b, torch::Tensor o, float s1, float s2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_mish", &fused_conv_mish, "Fused Convolution, Subtraction, and Mish");
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
    batch, _, h, w = x.shape
    k = conv_weight.shape[2]
    out_h, out_w = h - k + 1, w - k + 1
    out = torch.empty((batch, conv_weight.size(0), out_h, out_w), device=x.device)
    fused_ext.fused_conv_mish(x, conv_weight, conv_bias, out, subtract_value_1, subtract_value_2)
    return out
