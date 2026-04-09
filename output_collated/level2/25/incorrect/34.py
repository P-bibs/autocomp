# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_084802/code_8.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x, const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ output, int N, int C_in, int C_out, int H, int W, int K,
    int stride, int padding, int OH, int OW) {

    int n = blockIdx.z;
    int oh = blockIdx.y;
    int ow = blockIdx.x;
    
    // Each thread within the block processes one output channel (co)
    int co = threadIdx.x;
    if (co >= C_out) return;

    float sum = bias[co];
    
    // Perform convolution
    for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float val = x[((n * C_in + ci) * H + ih) * W + iw];
                    float w = weight[(((co * C_in + ci) * K + kh) * K + kw)];
                    sum += val * w;
                }
            }
        }
    }

    // Store in shared memory for reduction after sync
    extern __shared__ float s_data[];
    s_data[co] = sum;
    __syncthreads();

    // Perform min reduction across the channel dimension
    // Only one thread per (n, oh, ow) block performs the final activation
    if (co == 0) {
        float min_val = s_data[0];
        for (int i = 1; i < C_out; ++i) {
            if (s_data[i] < min_val) min_val = s_data[i];
        }
        float t = tanhf(min_val);
        output[((n * OH + oh) * OW + ow)] = tanhf(t);
    }
}

void launch_fused_kernel(const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& bias,
                         torch::Tensor& output, int stride, int padding) {
    int N = x.size(0); int C_in = x.size(1); int H = x.size(2); int W = x.size(3);
    int C_out = weight.size(0); int K = weight.size(2);
    int OH = (H + 2 * padding - K) / stride + 1;
    int OW = (W + 2 * padding - K) / stride + 1;

    dim3 block(C_out); // C_out is 64, well within max threads per block
    dim3 grid(OW, OH, N);
    
    size_t shared_mem = C_out * sizeof(float);
    fused_conv_min_tanh_kernel<<<grid, block, shared_mem>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C_in, C_out, H, W, K, stride, padding, OH, OW
    );
}
"""

cpp_source = r"""
void launch_fused_kernel(const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& bias,
                         torch::Tensor& output, int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused_kernel, "Fused Conv-Min-Tanh");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    N, C, H, W = x.shape
    C_out, _, K, _ = conv_weight.shape
    OH = (H + 2 * conv_padding - K) // conv_stride + 1
    OW = (W + 2 * conv_padding - K) // conv_stride + 1
    output = torch.empty((N, 1, OH, OW), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_weight, conv_bias, output, conv_stride, conv_padding)
    return output

def get_init_inputs(): return [16, 64, 3]
def get_inputs(): return [torch.rand(128, 16, 256, 256, device='cuda')]
