# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_095400/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'softmax_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

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
    # State for softmax (nn.Softmax)
    state_kwargs['softmax_dim'] = model.softmax.dim
    # State for sigmoid (nn.Sigmoid)
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

# Optimization: Custom fused CUDA kernel for ConvTranspose3d + Softmax + Sigmoid
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

// ConvTranspose3d kernel
__global__ void conv_transpose3d_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size, int in_ch, int out_ch,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int k_d, int k_h, int k_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_ch * out_d * out_h * out_w;
    
    if (out_idx >= total_elements) return;
    
    int tmp = out_idx;
    int w_out = tmp % out_w; tmp /= out_w;
    int h_out = tmp % out_h; tmp /= out_h;
    int d_out = tmp % out_d; tmp /= out_d;
    int c_out = tmp % out_ch; tmp /= out_ch;
    int n = tmp;
    
    float sum = 0.0f;
    int group = c_out * groups / out_ch;
    
    for (int kd = 0; kd < k_d; ++kd) {
        for (int kh = 0; kh < k_h; ++kh) {
            for (int kw = 0; kw < k_w; ++kw) {
                int d_in = d_out + pad_d - kd * dilation_d;
                int h_in = h_out + pad_h - kh * dilation_h;
                int w_in = w_out + pad_w - kw * dilation_w;
                
                if (d_in % stride_d == 0 && h_in % stride_h == 0 && w_in % stride_w == 0) {
                    d_in /= stride_d;
                    h_in /= stride_h;
                    w_in /= stride_w;
                    
                    if (d_in >= 0 && d_in < in_d &&
                        h_in >= 0 && h_in < in_h &&
                        w_in >= 0 && w_in < in_w) {
                        int c_in = (c_out * groups / out_ch) * (in_ch / groups) + 
                                   (kd * k_h * k_w + kh * k_w + kw) % (in_ch / groups);
                        
                        int input_idx = ((n * in_ch + c_in) * in_d + d_in) * in_h * in_w +
                                        h_in * in_w + w_in;
                        int weight_idx = ((c_out / (out_ch / groups)) * (in_ch / groups) + 
                                         (kd * k_h * k_w + kh * k_w + kw) % (in_ch / groups)) * 
                                         k_d * k_h * k_w + kd * k_h * k_w + kh * k_w + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    output[out_idx] = sum + bias[c_out];
}

// Fused Softmax + Sigmoid kernel
__global__ void fused_softmax_sigmoid_kernel(float* data, int num_elements, int dim_size, int outer_size, int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_size * inner_size) return;

    int outer = idx / inner_size;
    int inner = idx % inner_size;
    
    // Compute max for numerical stability
    float max_val = -1e20f;
    for (int i = 0; i < dim_size; ++i) {
        float val = data[(outer * dim_size + i) * inner_size + inner];
        if (val > max_val) max_val = val;
    }

    // Compute sum of exponentials
    float sum = 0.0f;
    for (int i = 0; i < dim_size; ++i) {
        float val = expf(data[(outer * dim_size + i) * inner_size + inner] - max_val);
        data[(outer * dim_size + i) * inner_size + inner] = val;
        sum += val;
    }

    // Normalize and Apply Sigmoid
    for (int i = 0; i < dim_size; ++i) {
        float sm = data[(outer * dim_size + i) * inner_size + inner] / sum;
        data[(outer * dim_size + i) * inner_size + inner] = 1.0f / (1.0f + expf(-sm));
    }
}

void run_conv_transpose3d_softmax_sigmoid(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups,
    int softmax_dim
) {
    // ConvTranspose3d parameters
    int batch_size = input.size(0);
    int in_ch = input.size(1);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);
    
    int out_ch = weight.size(1);
    int k_d = weight.size(2);
    int k_h = weight.size(3);
    int k_w = weight.size(4);
    
    int out_d = (in_d - 1) * stride_d - 2 * pad_d + dilation_d * (k_d - 1) + out_pad_d + 1;
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (k_h - 1) + out_pad_h + 1;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (k_w - 1) + out_pad_w + 1;
    
    // Launch ConvTranspose3d kernel
    int total_threads = batch_size * out_ch * out_d * out_h * out_w;
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    conv_transpose3d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_ch, out_ch, in_d, in_h, in_w,
        out_d, out_h, out_w, k_d, k_h, k_w,
        stride_d, stride_h, stride_w, pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w, dilation_d, dilation_h, dilation_w, groups
    );
    
    cudaDeviceSynchronize();
    
    // Launch fused Softmax+Sigmoid kernel
    int ndim = output.dim();
    int dim_size = output.size(softmax_dim);
    int outer_size = 1;
    for (int i = 0; i < softmax_dim; ++i) outer_size *= output.size(i);
    int inner_size = 1;
    for (int i = softmax_dim + 1; i < ndim; ++i) inner_size *= output.size(i);

    threads_per_block = 256;
    blocks = (outer_size * inner_size + threads_per_block - 1) / threads_per_block;
    fused_softmax_sigmoid_kernel<<<blocks, threads_per_block>>>(
        output.data_ptr<float>(), output.numel(), dim_size, outer_size, inner_size
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void run_conv_transpose3d_softmax_sigmoid(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups,
    int softmax_dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_conv_transpose3d_softmax_sigmoid", &run_conv_transpose3d_softmax_sigmoid, "ConvTranspose3D + Softmax + Sigmoid fusion");
}
"""

# Compile the extension
fused_module = load_inline(
    name='fused_conv_softmax_sigmoid',
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
    softmax_dim,
):
    # Compute output dimensions
    in_d, in_h, in_w = x.shape[2], x.shape[3], x.shape[4]
    k_d, k_h, k_w = conv_transpose_weight.shape[2], conv_transpose_weight.shape[3], conv_transpose_weight.shape[4]
    
    stride_d, stride_h, stride_w = conv_transpose_stride
    pad_d, pad_h, pad_w = conv_transpose_padding
    out_pad_d, out_pad_h, out_pad_w = conv_transpose_output_padding
    dilation_d, dilation_h, dilation_w = conv_transpose_dilation
    
    out_d = (in_d - 1) * stride_d - 2 * pad_d + dilation_d * (k_d - 1) + out_pad_d + 1
    out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (k_h - 1) + out_pad_h + 1
    out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (k_w - 1) + out_pad_w + 1
    
    out_ch = conv_transpose_weight.shape[1]
    batch_size = x.shape[0]
    
    # Create output tensor
    output = torch.empty(batch_size, out_ch, out_d, out_h, out_w, device=x.device, dtype=x.dtype)
    
    # Run the fused operation
    fused_module.run_conv_transpose3d_softmax_sigmoid(
        x, conv_transpose_weight, conv_transpose_bias, output,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w,
        dilation_d, dilation_h, dilation_w,
        conv_transpose_groups,
        softmax_dim
    )
    
    return output

# Test parameters
batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]
