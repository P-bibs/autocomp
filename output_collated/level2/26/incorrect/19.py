# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_041736/code_6.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------------------
# 1.  Vectorised CUDA kernel for fused ConvTranspose3D + Add + HardSwish
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float hardswish(float x) {
    float r = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * r * 0.16666667f;
}

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int out_depth,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding) {
    
    // Calculate output indices
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch_size * out_channels * out_depth * out_height * out_width) return;
    
    int tmp = out_idx;
    const int w_out = tmp % out_width; tmp /= out_width;
    const int h_out = tmp % out_height; tmp /= out_height;
    const int d_out = tmp % out_depth; tmp /= out_depth;
    const int c_out = tmp % out_channels; tmp /= out_channels;
    const int b = tmp;
    
    float sum = 0.0f;
    
    // Calculate input position
    const int d_in_start = (d_out + padding - kernel_size + 1 + stride - 1) / stride;
    const int d_in_end = min((d_out + padding) / stride + 1, in_depth);
    
    const int h_in_start = (h_out + padding - kernel_size + 1 + stride - 1) / stride;
    const int h_in_end = min((h_out + padding) / stride + 1, in_height);
    
    const int w_in_start = (w_out + padding - kernel_size + 1 + stride - 1) / stride;
    const int w_in_end = min((w_out + padding) / stride + 1, in_width);
    
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = max(0, d_in_start); kd < min(in_depth, d_in_end); ++kd) {
            for (int kh = max(0, h_in_start); kh < min(in_height, h_in_end); ++kh) {
                for (int kw = max(0, w_in_start); kw < min(in_width, w_in_end); ++kw) {
                    const int kd_kernel = d_out + padding - kd * stride;
                    const int kh_kernel = h_out + padding - kh * stride;
                    const int kw_kernel = w_out + padding - kw * stride;
                    
                    if (kd_kernel >= 0 && kd_kernel < kernel_size &&
                        kh_kernel >= 0 && kh_kernel < kernel_size &&
                        kw_kernel >= 0 && kw_kernel < kernel_size) {
                        
                        const int input_idx = b * (in_channels * in_depth * in_height * in_width) +
                                             c_in * (in_depth * in_height * in_width) +
                                             kd * (in_height * in_width) +
                                             kh * in_width + kw;
                        
                        const int weight_idx = c_out * (in_channels * kernel_size * kernel_size * kernel_size) +
                                              c_in * (kernel_size * kernel_size * kernel_size) +
                                              kd_kernel * (kernel_size * kernel_size) +
                                              kh_kernel * kernel_size + kw_kernel;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    sum += bias[c_out];
    output[out_idx] = sum;
}

__global__ void fused_add_hardswish_kernel(
    const float* __restrict__ conv_out,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    const int numel) {

    // each thread handles a vector of 4 floats
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;

    if (vec_idx >= numel) return;             // nothing to do

    // load 4 elements at once
    float4 c = reinterpret_cast<const float4*>(conv_out)[idx];
    float4 a = reinterpret_cast<const float4*>(add_input)[idx];

    // component-wise addition + hardswish
    float4 res;
    res.x = hardswish(c.x + a.x);
    res.y = hardswish(c.y + a.y);
    res.z = hardswish(c.z + a.z);
    res.w = hardswish(c.w + a.w);

    // store result
    if (vec_idx + 3 < numel) {
        reinterpret_cast<float4*>(output)[idx] = res;
    } else {
        if (vec_idx     < numel) output[vec_idx]     = res.x;
        if (vec_idx + 1 < numel) output[vec_idx + 1] = res.y;
        if (vec_idx + 2 < numel) output[vec_idx + 2] = res.z;
        if (vec_idx + 3 < numel) output[vec_idx + 3] = res.w;
    }
}

void launch_conv_transpose3d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    const int stride,
    const int padding) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);
    
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int out_depth = output.size(2);
    const int out_height = output.size(3);
    const int out_width = output.size(4);
    
    const int numel = batch_size * out_channels * out_depth * out_height * out_width;
    const dim3 block(256);
    const dim3 grid((numel + block.x - 1) / block.x);
    
    conv_transpose3d_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        out_depth,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "Conv transpose kernel failed: ", cudaGetErrorString(cudaGetLastError()));
}

void launch_fused_add_hardswish(
    const at::Tensor& conv_out,
    const at::Tensor& add_input,
    at::Tensor& output) {

    const int numel = conv_out.numel();
    const int vec_cnt = (numel + 3) / 4;               // number of float4 elements
    const dim3 block(256);
    const dim3 grid((vec_cnt + block.x - 1) / block.x);

    fused_add_hardswish_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        conv_out.data_ptr<float>(),
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "Fused add+hardswish kernel failed: ", cudaGetErrorString(cudaGetLastError()));
}
"""

# -------------------------------------------------------------------------
# 2.  C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void launch_conv_transpose3d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    const int stride,
    const int padding);

void launch_fused_add_hardswish(
    const at::Tensor& conv_out,
    const at::Tensor& add_input,
    at::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d", &launch_conv_transpose3d,
          "Custom ConvTranspose3D kernel");
    m.def("fused_add_hardswish", &launch_fused_add_hardswish,
          "Fused add + HardSwish (vectorised)");
}
"""

# -------------------------------------------------------------------------
# 3.  Compile the inline extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op_vec',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# 4.  Functional model – the only symbol that will be imported
# -------------------------------------------------------------------------
def functional_model(
    x,
    add_input,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    bias,
):
    """
    Custom ConvTranspose3D followed by element-wise addition and HardSwish.
    Both operations are performed by vectorised CUDA kernels.
    """
    # Ensure the tensors are contiguous
    x = x.contiguous()
    add_input = add_input.contiguous()
    conv_transpose_weight = conv_transpose_weight.contiguous()
    conv_transpose_bias = conv_transpose_bias.contiguous()
    
    # Calculate output dimensions for conv transpose
    batch_size, in_channels, in_depth, in_height, in_width = x.shape
    out_channels = conv_transpose_weight.size(0)
    kernel_size = conv_transpose_weight.size(2)
    
    # Calculate output dimensions
    out_depth = (in_depth - 1) * conv_transpose_stride + kernel_size - 2 * conv_transpose_padding + conv_transpose_output_padding
    out_height = (in_height - 1) * conv_transpose_stride + kernel_size - 2 * conv_transpose_padding + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride + kernel_size - 2 * conv_transpose_padding + conv_transpose_output_padding
    
    # Allocate output tensor for conv transpose
    conv_out = torch.empty(batch_size, out_channels, out_depth, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Launch custom conv transpose kernel
    fused_ext.conv_transpose3d(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        conv_out,
        conv_transpose_stride,
        conv_transpose_padding
    )
    
    # Allocate final output tensor
    output = torch.empty_like(conv_out)
    
    # Launch the vectorised fused kernel for add + hardswish
    fused_ext.fused_add_hardswish(conv_out, add_input, output)
    
    return output

# -------------------------------------------------------------------------
# 5.  Minimal test (optional – not required for evaluation)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Same configuration as in the original script
    batch_size = 128
    in_channels = 32
    out_channels = 64
    D = H = W = 16
    kernel_size = 3
    stride = 2
    padding = 1
    output_padding = 1

    # Build parameters on the GPU
    x = torch.rand(batch_size, in_channels, D, H, W, device='cuda')
    add_input = torch.rand(batch_size, out_channels,
                           D * stride, H * stride, W * stride, device='cuda')

    params = {
        "conv_transpose_weight": torch.randn(out_channels, in_channels,
                                             kernel_size, kernel_size, kernel_size,
                                             device='cuda'),
        "conv_transpose_bias": torch.randn(out_channels, device='cuda'),
        "conv_transpose_stride": stride,
        "conv_transpose_padding": padding,
        "conv_transpose_output_padding": output_padding,
        "conv_transpose_groups": 1,
        "conv_transpose_dilation": 1,
        "bias": torch.randn(out_channels, 1, 1, 1, 1, device='cuda'),
    }

    # Warm-up
    with torch.no_grad():
        _ = functional_model(x, add_input, **params)

    # Benchmark
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        for _ in range(10):
            y = functional_model(x, add_input, **params)
    end.record()
    torch.cuda.synchronize()

    print(f"Average time per iteration: {start.elapsed_time(end) / 10:.3f} ms")
    print(f"Output shape: {y.shape}")
