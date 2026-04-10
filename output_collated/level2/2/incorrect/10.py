# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_162535/code_4.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape', 'scaling_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias', 'scaling_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

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
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    if 'scaling_factor' in flat_state:
        state_kwargs['scaling_factor'] = flat_state['scaling_factor']
    else:
        state_kwargs['scaling_factor'] = getattr(model, 'scaling_factor')
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

# --- CUDA Kernel ---
# Implementation: Custom ConvTranspose2d (Naive Im2Col-like structure) 
# fused with bias, clamp, scale operations in one pass to maximize L1/registers usage.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t>
__global__ void fused_convtranspose2d_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> bias,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output,
    const int in_c, const int out_c, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int out_h, const int out_w, const float scaling_factor
) {
    int b = blockIdx.z;
    int oc = blockIdx.y;
    int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h_idx = blockIdx.y * blockDim.y + threadIdx.y; // Logic mapped carefully

    // Map output coordinates
    int out_y = blockIdx.y;
    int out_x = threadIdx.x;
    
    if (out_y >= out_h || out_x >= out_w) return;

    scalar_t val = bias[oc];
    
    // Transposed Convolution loop
    for (int ic = 0; ic < in_c; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int ih = (out_y + pad_h - kh);
                int iw = (out_x + pad_w - kw);
                
                if (ih % stride_h == 0 && iw % stride_w == 0) {
                    ih /= stride_h;
                    iw /= stride_w;
                    if (ih >= 0 && ih < input.size(2) && iw >= 0 && iw < input.size(3)) {
                        val += input[b][ic][ih][iw] * weight[ic][oc][kh][kw];
                    }
                }
            }
        }
    }

    // Fused elementwise ops: Clamp -> Scale -> Clamp -> Divide
    val = fmaxf(0.0f, fminf(1.0f, val));
    val *= (scalar_t)scaling_factor;
    val = fmaxf(0.0f, fminf(1.0f, val));
    val /= (scalar_t)scaling_factor;

    output[b][oc][out_y][out_x] = val;
}

void fused_op_forward(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::Tensor& output, int stride, int padding, float scaling_factor
) {
    const int batch = input.size(0);
    const int oc = weight.size(1);
    const int oh = output.size(2);
    const int ow = output.size(3);

    dim3 blocks(ow, oh);
    dim3 threads(1, 1); // Simplification for grid dimensions
    
    // Note: In production this would use tiling/shared memory.
    // Given the constraints, we implement standard direct convolution.
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_conv", ([&] {
        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < oc; ++c) {
                // Wrapper trigger
                fused_convtranspose2d_kernel<scalar_t><<<dim3(ow/16+1, oh/16+1), dim3(16, 16)>>>(
                    input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    weight.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                    output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    input.size(1), oc, weight.size(2), weight.size(3),
                    stride, stride, padding, padding, oh, ow, scaling_factor
                );
            }
        }
    }));
}
"""

cpp_source = """
#include <torch/extension.h>
void fused_op_forward(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, at::Tensor& output, int stride, int padding, float scaling_factor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused_op", &fused_op_forward, "Fused Op"); }
"""

fused_ext = load_inline(name='fused_ct', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
                       extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, 
                     conv_transpose_groups, conv_transpose_dilation, bias, scaling_factor):
    # Weight shape for kernel: [in_channels, out_channels, kH, kW]
    w = conv_transpose_weight.permute(1, 0, 2, 3) 
    oh = (x.shape[2] - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + conv_transpose_weight.shape[2] + conv_transpose_output_padding[0]
    ow = (x.shape[3] - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + conv_transpose_weight.shape[3] + conv_transpose_output_padding[1]
    out = torch.empty((x.shape[0], w.size(1), oh, ow), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, w, conv_transpose_bias.flatten(), out, conv_transpose_stride[0], conv_transpose_padding[0], float(scaling_factor))
    return out

# Placeholders as required by interface
batch_size, in_channels, out_channels, height, width = 128, 64, 64, 128, 128
kernel_size, stride, padding, output_padding, bias_shape, scaling_factor = 3, (2, 2), (1, 1), (1, 1), (64, 1, 1), 2.0
def get_init_inputs(): return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]
def get_inputs(): return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
