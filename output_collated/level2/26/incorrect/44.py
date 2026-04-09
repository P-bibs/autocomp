# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_045050/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x', 'add_input']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
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

# Optimized implementation with implicit GEMM fused with add + HardSwish
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float hardswish_impl(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) * 0.16666667f;
}

__global__ void fused_conv_tr_add_hardswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    int N, int IC, int OC, int ID, int IH, int IW, 
    int KD, int KH, int KW,
    int OD, int OH, int OW, 
    int stride, int padding) {

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= N * OC * OD * OH * OW) return;

    // Decompose linear index to 5D output coordinates
    int tmp = out_idx;
    int ow = tmp % OW; tmp /= OW;
    int oh = tmp % OH; tmp /= OH;
    int od = tmp % OD; tmp /= OD;
    int oc = tmp % OC; tmp /= OC;
    int n  = tmp;

    float acc = bias[oc];
    
    // Transposed convolution accumulation
    for (int ic = 0; ic < IC; ++ic) {
        for (int kd = 0; kd < KD; ++kd) {
            int id = od * stride - padding + kd;
            if (id >= 0 && id < ID) {
                for (int kh = 0; kh < KH; ++kh) {
                    int ih = oh * stride - padding + kh;
                    if (ih >= 0 && ih < IH) {
                        for (int kw = 0; kw < KW; ++kw) {
                            int iw = ow * stride - padding + kw;
                            if (iw >= 0 && iw < IW) {
                                // Input tensor access pattern: [N, IC, ID, IH, IW]
                                float in_val = input[(((n * IC + ic) * ID + id) * IH + ih) * IW + iw];
                                // Weight tensor access pattern: [OC, IC, KD, KH, KW] 
                                float wt_val = weight[(((oc * IC + ic) * KD + kd) * KH + kh) * KW + kw];
                                acc += in_val * wt_val;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Fused add + hardswish
    float x = acc + add_input[out_idx];
    output[out_idx] = hardswish_impl(x);
}

void launch_fused_op(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& add_input,
    at::Tensor& output,
    int stride,
    int padding) {
    
    int N = input.size(0);
    int IC = input.size(1);
    int ID = input.size(2);
    int IH = input.size(3);
    int IW = input.size(4);
    
    int OC = weight.size(0);  // Note: weight layout is [OC, IC, KD, KH, KW] for transposed conv
    
    int KD = weight.size(2);
    int KH = weight.size(3);
    int KW = weight.size(4);
    
    int OD = output.size(2);
    int OH = output.size(3);
    int OW = output.size(4);
    
    const int numel = output.numel();
    const dim3 block(256);
    const dim3 grid((numel + block.x - 1) / block.x);
    
    fused_conv_tr_add_hardswish_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, IC, OC, ID, IH, IW,
        KD, KH, KW,
        OD, OH, OW,
        stride, padding
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "CUDA kernel failed: ", cudaGetErrorString(cudaGetLastError()));
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_fused_op(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& add_input,
    at::Tensor& output,
    int stride,
    int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused_op, "Fused ConvTranspose3d + Add + HardSwish");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    add_input,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    **kwargs
):
    # Allocate output tensor
    output = torch.empty_like(add_input)
    
    # Launch fused kernel
    fused_ext.fused_op(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        add_input, 
        output, 
        conv_transpose_stride, 
        conv_transpose_padding
    )
    
    return output
