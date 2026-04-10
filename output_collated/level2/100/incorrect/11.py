# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_115141/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'min_value', 'divisor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'min_value', 'divisor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

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
    if 'min_value' in flat_state:
        state_kwargs['min_value'] = flat_state['min_value']
    else:
        state_kwargs['min_value'] = getattr(model, 'min_value')
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
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
# CUDA source – the fused conv‑transpose + clamp + scale kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int stride, const int padding,
    const int dilation,
    const int K, const float min_val, const float divisor)
{
    extern __shared__ float s_weight[];
    
    // Each block handles blockDim.y output channels (oc_start to oc_start + blockDim.y - 1)
    const int block_y = blockDim.y;
    const int oc_start = blockIdx.y * block_y;
    const int weight_per_oc = C_in * K * K * K;
    
    // Load weight slice into shared memory
    for (int tid = threadIdx.x + threadIdx.y * blockDim.x; 
         tid < block_y * weight_per_oc; 
         tid += blockDim.x * blockDim.y) {
        int oc_idx = tid / weight_per_oc;
        int rem = tid % weight_per_oc;
        // Weight is stored as (C_in, C_out, K, K, K)
        // Original layout: in_c * (C_out * K^3) + out_c * K^3 + k_idx
        int ic = rem / (K * K * K);
        int k_idx = rem % (K * K * K);
        int oc = oc_start + oc_idx;
        if (oc < C_out) {
            s_weight[tid] = weight[ic * (C_out * K * K * K) + oc * (K * K * K) + k_idx];
        }
    }
    __syncthreads();

    // Determine output coordinate
    const long long linear_out = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (linear_out >= (long long)N * D_out * H_out * W_out) return;

    int remaining = linear_out;
    int ow = remaining % W_out; remaining /= W_out;
    int oh = remaining % H_out; remaining /= H_out;
    int od = remaining % D_out; remaining /= D_out;
    int n = remaining;

    int oc = oc_start + threadIdx.y;
    if (oc >= C_out) return;

    float sum = 0.0f;
    // Transposed Conv calculation
    for (int kd = 0; kd < K; ++kd) {
        int id_raw = od + padding - kd * dilation;
        if (id_raw < 0 || id_raw % stride != 0) continue;
        int id = id_raw / stride;
        if (id >= D_in) continue;

        for (int kh = 0; kh < K; ++kh) {
            int ih_raw = oh + padding - kh * dilation;
            if (ih_raw < 0 || ih_raw % stride != 0) continue;
            int ih = ih_raw / stride;
            if (ih >= H_in) continue;

            for (int kw = 0; kw < K; ++kw) {
                int iw_raw = ow + padding - kw * dilation;
                if (iw_raw < 0 || iw_raw % stride != 0) continue;
                int iw = iw_raw / stride;
                if (iw >= W_in) continue;

                for (int ic = 0; ic < C_in; ++ic) {
                    float in_val = input[((n * C_in + ic) * D_in * H_in + id * H_in + ih) * W_in + iw];
                    float w_val = s_weight[(ic * block_y + threadIdx.y) * (K * K * K) + (kd * K * K + kh * K + kw)];
                    sum += in_val * w_val;
                }
            }
        }
    }

    if (bias != nullptr) sum += bias[oc];
    output[((n * C_out + oc) * D_out * H_out + od * H_out + oh) * W_out + ow] = fmaxf(sum, min_val) / divisor;
}

void fused_conv_transpose(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, 
                          torch::Tensor& output, int stride, int padding, int dilation, int K, float min_val, float divisor) {
    const int N = input.size(0), C_in = input.size(1), D_in = input.size(2), H_in = input.size(3), W_in = input.size(4);
    const int C_out = weight.size(1), D_out = output.size(2), H_out = output.size(3), W_out = output.size(4);
    
    dim3 block(128, 4);
    dim3 grid((N * D_out * H_out * W_out + block.x - 1) / block.x, (C_out + block.y - 1) / block.y);
    size_t smem = block.y * C_in * K * K * K * sizeof(float);
    
    conv_transpose_fused_kernel<<<grid, block, smem>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.numel() ? bias.data_ptr<float>() : nullptr, output.data_ptr<float>(),
        N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
        stride, padding, dilation, K, min_val, divisor);
}
"""

cpp_source = r"""
void fused_conv_transpose(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, 
                          torch::Tensor& output, int stride, int padding, int dilation, int K, float min_val, float divisor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused_conv_transpose", &fused_conv_transpose); }
"""

fused_ext = load_inline(name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source, 
                        extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, min_value, divisor):
    K = conv_transpose_weight.size(2)
    D_out = (x.size(2) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    H_out = (x.size(3) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    W_out = (x.size(4) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    out = torch.empty((x.size(0), conv_transpose_weight.size(1), D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    fused_ext.fused_conv_transpose(x, conv_transpose_weight, conv_transpose_bias if conv_transpose_bias is not None else torch.empty(0, device=x.device), 
                                   out, conv_transpose_stride, conv_transpose_padding, conv_transpose_dilation, K, min_value, divisor)
    return out

batch_size, in_channels, out_channels = 16, 64, 128
depth, height, width, kernel_size, stride, padding, min_value, divisor = 24, 48, 48, 3, 2, 1, -1.0, 2.0

def get_init_inputs(): return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]
def get_inputs(): return [torch.rand(batch_size, in_channels, depth, height, width)]
