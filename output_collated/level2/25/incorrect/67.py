# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_091623/code_14.py
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

# -------------------------------------------------------------------------
#  CUDA source – Fused Kernel
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out,
    int H, int W, int K,
    int stride, int padding,
    int OH, int OW)
{
    extern __shared__ float weight_shared[];
    int total_weights = C_out * C_in * K * K;
    for (int i = threadIdx.x; i < total_weights; i += blockDim.x) {
        weight_shared[i] = weight[i];
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * OH * OW;

    if (tid < total_elements) {
        int n = tid / (OH * OW);
        int rem = tid % (OH * OW);
        int oh = rem / OW;
        int ow = rem % OW;

        float min_val = 1e30f; // Sufficiently large

        for (int co = 0; co < C_out; ++co) {
            float sum = bias[co];
            int co_offset = co * C_in * K * K;

            for (int ci = 0; ci < C_in; ++ci) {
                int ci_offset = ci * K * K;
                int x_base = ((n * C_in + ci) * H);

                for (int kh = 0; kh < K; ++kh) {
                    int ih = oh * stride + kh - padding;
                    if (ih < 0 || ih >= H) continue;
                    
                    int x_row = (x_base + ih) * W;
                    for (int kw = 0; kw < K; ++kw) {
                        int iw = ow * stride + kw - padding;
                        if (iw < 0 || iw >= W) continue;
                        
                        sum += __ldg(&x[x_row + iw]) * weight_shared[co_offset + ci_offset + kh * K + kw];
                    }
                }
            }
            if (sum < min_val) min_val = sum;
        }

        float t = tanhf(min_val);
        output[tid] = tanhf(t);
    }
}

void fused_op_forward(
    const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& bias,
    torch::Tensor& output, int stride, int padding) 
{
    int N = x.size(0), C_in = x.size(1), H = x.size(2), W = x.size(3);
    int C_out = weight.size(0), K = weight.size(2);
    int OH = (H + 2 * padding - K) / stride + 1;
    int OW = (W + 2 * padding - K) / stride + 1;

    int threads = 256;
    int blocks = (N * OH * OW + threads - 1) / threads;
    size_t shared_size = C_out * C_in * K * K * sizeof(float);

    fused_conv_min_tanh_kernel<<<blocks, threads, shared_size>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C_in, C_out, H, W, K, stride, padding, OH, OW
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& bias, 
                      torch::Tensor& output, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv-Min-Tanh");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    N, C_in, H, W = x.shape
    C_out, _, K, _ = conv_weight.shape
    OH = (H + 2 * conv_padding - K) // conv_stride + 1
    OW = (W + 2 * conv_padding - K) // conv_stride + 1
    
    output = torch.empty((N, 1, OH, OW), device='cuda', dtype=torch.float32)
    fused_ext.fused_op(x, conv_weight, conv_bias, output, conv_stride, conv_padding)
    return output
