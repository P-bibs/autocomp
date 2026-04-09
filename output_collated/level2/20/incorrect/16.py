# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_130013/code_13.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by a sum, 
    a residual add, a multiplication, and another residual add.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

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
    # State for conv_transpose (nn.ConvTranspose3d)
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
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
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

# We implement a custom CUDA kernel that handles the Transposed Convolution (Weight-Accumulated GEMM)
# and the fused element-wise post-processing arithmetic in a single pass to maximize L1/L2 cache locality.
# Using a direct-addressing approach for the transposed convolution (scatter-add pattern).

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_tr_arith_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ post_bias,
    float* __restrict__ output,
    int64_t B, int64_t IC, int64_t OC,
    int64_t ID, int64_t IH, int64_t IW,
    int64_t KD, int64_t KH, int64_t KW,
    int64_t OD, int64_t OH, int64_t OW,
    int64_t stride, int64_t padding
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = B * OC * OD * OH * OW;
    if (idx >= total) return;

    // Decode linear index to output tensor coordinates (N, C, D, H, W)
    int64_t tmp = idx;
    int64_t w = tmp % OW; tmp /= OW;
    int64_t h = tmp % OH; tmp /= OH;
    int64_t d = tmp % OD; tmp /= OD;
    int64_t c = tmp % OC; tmp /= OC;
    int64_t n = tmp;

    float acc = conv_bias[c];

    // Transposed Convolution: Sliding window logic
    for (int64_t ki = 0; ki < KD; ++ki) {
        for (int64_t kj = 0; kj < KH; ++kj) {
            for (int64_t kk = 0; kk < KW; ++kk) {
                int64_t id = (d + padding - ki);
                int64_t ih = (h + padding - kj);
                int64_t iw = (w + padding - kk);

                if (id % stride == 0 && ih % stride == 0 && iw % stride == 0) {
                    id /= stride; ih /= stride; iw /= stride;
                    if (id >= 0 && id < ID && ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                        for (int64_t ic = 0; ic < IC; ++ic) {
                            float in_val = input[n * (IC * ID * IH * IW) + ic * (ID * IH * IW) + id * (IH * IW) + ih * IW + iw];
                            float w_val = weight[ic * (OC * KD * KH * KW) + c * (KD * KH * KW) + ki * (KH * KW) + kj * KW + kk];
                            acc += in_val * w_val;
                        }
                    }
                }
            }
        }
    }

    // Fused element-wise arithmetic: result = ((x + bias) + x) * x + x = 2*x^2 + x*bias + x
    float b_val = post_bias[c];
    float result = ((acc + b_val) + acc) * acc + acc;
    output[idx] = result;
}

void fused_op_forward(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& conv_bias, 
    const torch::Tensor& post_bias, torch::Tensor& output,
    int64_t stride, int64_t padding
) {
    int64_t B = input.size(0), IC = input.size(1), ID = input.size(2), IH = input.size(3), IW = input.size(4);
    int64_t OC = weight.size(1), KD = weight.size(2), KH = weight.size(3), KW = weight.size(4);
    int64_t OD = output.size(2), OH = output.size(3), OW = output.size(4);
    
    int64_t total = B * OC * OD * OH * OW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    fused_conv_tr_arith_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), conv_bias.data_ptr<float>(), 
        post_bias.data_ptr<float>(), output.data_ptr<float>(),
        B, IC, OC, ID, IH, IW, KD, KH, KW, OD, OH, OW, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& conv_bias, 
                     const torch::Tensor& post_bias, torch::Tensor& output, int64_t stride, int64_t padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Transposed Conv + Arith");
}
"""

fused_ext = load_inline(name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
                       extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    # Shape logic matching conv_transpose3d
    N, C_in, D_in, H_in, W_in = x.shape
    C_out, _, KD, KH, KW = conv_transpose_weight.shape
    stride, padding = conv_transpose_stride[0], conv_transpose_padding[0]
    
    D_out = (D_in - 1) * stride - 2 * padding + KD + conv_transpose_output_padding[0]
    H_out = (H_in - 1) * stride - 2 * padding + KH + conv_transpose_output_padding[1]
    W_out = (W_in - 1) * stride - 2 * padding + KW + conv_transpose_output_padding[2]
    
    output = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, bias.view(-1), output, stride, padding)
    return output
