# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_084802/code_10.py
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

# The custom CUDA kernel implements a fused convolution, channel-wise min reduction,
# and double tanh activation. We map blocks to (N, OH, OW), with threads processing 
# the C_out dimension.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x, const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ output, int N, int C_in, int C_out, int H, int W, int K,
    int OH, int OW) {

    int n = blockIdx.z;
    int oh = blockIdx.y;
    int ow = blockIdx.x;
    int tid = threadIdx.x;

    // Use shared memory for weight caching to improve performance on 2080Ti
    // Storing a tiles of weight is complex; for this implementation we focus on 
    // vectorized register accumulation.
    float local_sum = 0.0f;
    
    // Each thread computes one output channel index if threads <= C_out 
    // and performs full reduction across the spatial/input dimension.
    if (tid < C_out) {
        local_sum = bias[tid];
        for (int ci = 0; ci < C_in; ++ci) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int ih = oh + kh;
                    int iw = ow + kw;
                    float val = x[((n * C_in + ci) * H + ih) * W + iw];
                    float w = weight[(((tid * C_in + ci) * K + kh) * K + kw)];
                    local_sum += val * w;
                }
            }
        }
    }

    // Shared buffer for reduction
    extern __shared__ float s_data[];
    s_data[tid] = (tid < C_out) ? local_sum : 1e30f; // Max float
    __syncthreads();

    // Min reduction across all output channels
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (tid + s) < C_out) {
            s_data[tid] = fminf(s_data[tid], s_data[tid + s]);
        }
        __syncthreads();
    }

    // Final result Calculation
    if (tid == 0) {
        float min_val = s_data[0];
        output[((n * OH) + oh) * OW + ow] = tanhf(tanhf(min_val));
    }
}

void launch_fused_conv(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor& out) {
    int N = x.size(0), C_in = x.size(1), H = x.size(2), W = x.size(3);
    int C_out = weight.size(0), K = weight.size(2);
    int OH = H - K + 1, OW = W - K + 1;
    
    dim3 grid(OW, OH, N);
    int threads_per_block = ((C_out + 31) / 32) * 32;
    size_t shared_mem = threads_per_block * sizeof(float);

    fused_conv_min_tanh_kernel<<<grid, threads_per_block, shared_mem>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), N, C_in, C_out, H, W, K, OH, OW);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_fused_conv(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor& out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_conv", &launch_fused_conv, "Fused Conv/Min/Tanh kernel");
}
"""

fused_ext = load_inline(
    name='fused_conv_ext', 
    cpp_sources=cpp_source, 
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], 
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1):
    # Enforce constraints defined for the optimized kernel
    assert conv_stride == 1 and conv_padding == 0 and conv_dilation == 1 and conv_groups == 1
    N, C_in, H, W = x.shape
    C_out, _, K, _ = conv_weight.shape
    out = torch.empty((N, 1, H - K + 1, W - K + 1), device=x.device)
    fused_ext.launch_fused_conv(x, conv_weight, conv_bias, out)
    return out
