# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_133559/code_7.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_post_conv_kernel(
    const float4* __restrict__ input,
    const float* __restrict__ bias,
    float4* __restrict__ output,
    const int num_elements_float4,
    const int spatial_size,
    const int out_channels,
    const int total_elements
) {
    extern __shared__ float bias_shared[];
    
    // Load bias into shared memory with grid-stride loop for robustness
    for (int i = threadIdx.x; i < out_channels; i += blockDim.x) {
        bias_shared[i] = bias[i];
    }
    __syncthreads();

    // Grid-stride loop: each thread processes multiple float4s
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < num_elements_float4; 
         idx += stride) {
        
        const int elem_base = idx * 4;
        const int channel = (elem_base / spatial_size) % out_channels;
        const float b = bias_shared[channel];
        const float4 x_vec = input[idx];

        // Inline processing with fast math and masking
        const float two = 2.0f;
        const float one = 1.0f;

        float4 res;
        res.x = (elem_base < total_elements) ? 
                __fmul_rn(x_vec.x, __fadd_rn(__fmul_rn(two, x_vec.x), __fadd_rn(b, one))) : 0.0f;
        res.y = (elem_base + 1 < total_elements) ? 
                __fmul_rn(x_vec.y, __fadd_rn(__fmul_rn(two, x_vec.y), __fadd_rn(b, one))) : 0.0f;
        res.z = (elem_base + 2 < total_elements) ? 
                __fmul_rn(x_vec.z, __fadd_rn(__fmul_rn(two, x_vec.z), __fadd_rn(b, one))) : 0.0f;
        res.w = (elem_base + 3 < total_elements) ? 
                __fmul_rn(x_vec.w, __fadd_rn(__fmul_rn(two, x_vec.w), __fadd_rn(b, one))) : 0.0f;

        output[idx] = res;
    }
}

// Conv transpose 3D kernel using optimized memory access
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
    const int padding,
    const int output_padding
) {
    extern __shared__ float weight_shared[];
    
    // Load weight into shared memory
    const int weight_size = in_channels * out_channels * kernel_size * kernel_size * kernel_size;
    for (int i = threadIdx.x; i < weight_size; i += blockDim.x) {
        weight_shared[i] = weight[i];
    }
    __syncthreads();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;

    for (int idx = tid; idx < total_elements; idx += total_threads) {
        const int w = idx % out_width;
        const int h = (idx / out_width) % out_height;
        const int d = (idx / (out_width * out_height)) % out_depth;
        const int oc = (idx / (out_width * out_height * out_depth)) % out_channels;
        const int b = idx / (out_width * out_height * out_depth * out_channels);

        float value = (bias != nullptr) ? bias[oc] : 0.0f;

        // Compute input coordinates
        for (int kd = 0; kd < kernel_size; ++kd) {
            const int id = d - kd * stride + padding;
            if (id < 0 || id >= in_depth * stride) continue;
            if (id % stride != 0) continue;
            const int in_d = id / stride;

            for (int kh = 0; kh < kernel_size; ++kh) {
                const int ih = h - kh * stride + padding;
                if (ih < 0 || ih >= in_height * stride) continue;
                if (ih % stride != 0) continue;
                const int in_h = ih / stride;

                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int iw = w - kw * stride + padding;
                    if (iw < 0 || iw >= in_width * stride) continue;
                    if (iw % stride != 0) continue;
                    const int in_w = iw / stride;

                    for (int ic = 0; ic < in_channels; ++ic) {
                        const int weight_idx = ((ic * out_channels + oc) * kernel_size + kd) * kernel_size * kernel_size + 
                                               kh * kernel_size + kw;
                        const int input_idx = ((b * in_channels + ic) * in_depth + in_d) * in_height * in_width + 
                                              in_h * in_width + in_w;
                        value += input[input_idx] * weight_shared[weight_idx];
                    }
                }
            }
        }

        output[idx] = value;
    }
}

void fused_post_conv_forward(torch::Tensor input, torch::Tensor bias, torch::Tensor output) {
    const int total_elements = input.numel();
    const int num_elements_float4 = (total_elements + 3) / 4;
    const int spatial_size = input.size(2) * input.size(3) * input.size(4);
    const int out_channels = input.size(1);

    const int threads = 256;
    const int blocks = std::min((num_elements_float4 + threads - 1) / threads, 65535);
    
    fused_post_conv_kernel<<<blocks, threads, out_channels * sizeof(float)>>>(
        (const float4*)input.data_ptr<float>(),
        bias.data_ptr<float>(),
        (float4*)output.data_ptr<float>(),
        num_elements_float4, spatial_size, out_channels, total_elements
    );
}

void conv_transpose3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
) {
    // For simplicity, we assume groups=1 and dilation=1 as in the original
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);
    const int kernel_size = weight.size(2);
    
    const int out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    const int threads = 256;
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int blocks = std::min((total_elements + threads - 1) / threads, 65535);
    const int shared_mem = in_channels * out_channels * kernel_size * kernel_size * kernel_size * sizeof(float);

    float* bias_ptr = bias_opt.has_value() ? bias_opt.value().data_ptr<float>() : nullptr;
    
    conv_transpose3d_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
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
        padding,
        output_padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_post_conv_forward(torch::Tensor input, torch::Tensor bias, torch::Tensor output);
void conv_transpose3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv_forward", &fused_post_conv_forward, "Fused post-conv fwd");
    m.def("conv_transpose3d_forward", &conv_transpose3d_forward, "Conv transpose 3D fwd");
}
"""

module = load_inline(
    name='fused_ext', 
    cpp_sources=cpp_source, 
    cuda_sources=cuda_source, 
    extra_cuda_cflags=['-O3', '--use_fast_math'], 
    with_cuda=True
)

def functional_model(
    x,
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
    # Perform the convolution with custom CUDA kernel
    out_channels = conv_transpose_weight.size(1)
    kernel_size = conv_transpose_weight.size(2)
    
    # Compute output dimensions
    in_depth, in_height, in_width = x.shape[2], x.shape[3], x.shape[4]
    out_depth = (in_depth - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_height = (in_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    
    conv_output = torch.empty((x.size(0), out_channels, out_depth, out_height, out_width), device=x.device, dtype=x.dtype)
    
    module.conv_transpose3d_forward(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        conv_output,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_groups,
        conv_transpose_dilation
    )
    
    # Flatten bias for simplified kernel indexing
    bias_flat = bias.view(-1)
    
    # Use optimized fused kernel for the intensive post-processing element-wise ops
    final_output = torch.empty_like(conv_output)
    module.fused_post_conv_forward(conv_output, bias_flat, final_output)
    
    return final_output

# Test parameters
batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
