# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123254/code_4.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'bias_shape', 'stride', 'padding', 'output_padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
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

# Optimized CUDA kernel with tiling, shared memory, and register usage
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define KERN_SIZE 4

__global__ void fused_conv_transpose_bias_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int IC, int OC, int IH, int IW, int OH, int OW, int K,
    int stride, int pad) {
    
    extern __shared__ float shared_weight[];
    
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    int n  = blockIdx.y;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Load weight tile to shared memory
    for (int i = ty*TILE_SIZE + tx; i < IC*K*K; i += TILE_SIZE*TILE_SIZE) {
        if (i < IC * K * K) {
            shared_weight[i] = weight[i * OC + oc];
        }
    }
    __syncthreads();

    if (oc >= OC || n >= N) return;

    float bias_val = bias[oc];

    for (int oh = ty; oh < OH; oh += TILE_SIZE) {
        for (int ow = tx; ow < OW; ow += TILE_SIZE) {
            float sum = 0.f;
            
            int ih_base = oh - (K-1) + pad;
            int iw_base = ow - (K-1) + pad;
            
            for (int ic = 0; ic < IC; ++ic) {
                const float* w_ptr = &shared_weight[ic * K * K];
                for (int kh = 0; kh < K; ++kh) {
                    int ih = ih_base + kh;
                    if (ih % stride != 0) continue;
                    ih /= stride;
                    if (ih < 0 || ih >= IH) continue;
                    
                    for (int kw = 0; kw < K; ++kw) {
                        int iw = iw_base + kw;
                        if (iw % stride != 0) continue;
                        iw /= stride;
                        if (iw < 0 || iw >= IW) continue;
                        
                        float in_val = input[((n * IC + ic) * IH + ih) * IW + iw];
                        float w_val = w_ptr[kh * K + kw];
                        sum += in_val * w_val;
                    }
                }
            }
            
            // Apply bias and tanh
            int out_idx = ((n * OC + oc) * OH + oh) * OW + ow;
            if (oh < OH && ow < OW) {
                output[out_idx] = tanhf(sum - bias_val);
            }
        }
    }
}

void fused_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding) {
    
    int N  = input.size(0);
    int IC = input.size(1);
    int IH = input.size(2);
    int IW = input.size(3);
    int OC = weight.size(0);
    int K  = weight.size(2);
    
    int OH = (IH - 1) * stride - 2 * padding + K;
    int OW = (IW - 1) * stride - 2 * padding + K;
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((OC + TILE_SIZE - 1) / TILE_SIZE, N);
    
    size_t shared_mem_size = IC * K * K * sizeof(float);
    
    fused_conv_transpose_bias_tanh_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, IC, OC, IH, IW, OH, OW, K,
        stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
              torch::Tensor output, int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused Conv Transpose + Bias + Tanh");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_op',
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
    # Only support single group, no dilation, no output padding for this optimized version
    assert conv_transpose_groups == 1
    assert conv_transpose_dilation == (1, 1)
    assert conv_transpose_output_padding == (0, 0)
    assert isinstance(conv_transpose_stride, tuple) and len(conv_transpose_stride) == 2
    assert isinstance(conv_transpose_padding, tuple) and len(conv_transpose_padding) == 2
    
    stride_h, stride_w = conv_transpose_stride
    pad_h, pad_w = conv_transpose_padding
    
    # Assuming square kernels and symmetric parameters for simplicity
    assert stride_h == stride_w
    assert pad_h == pad_w
    
    N, IC, IH, IW = x.shape
    OC, _, K, _ = conv_transpose_weight.shape
    
    OH = (IH - 1) * stride_h - 2 * pad_h + K
    OW = (IW - 1) * stride_w - 2 * pad_w + K
    
    out = torch.empty((N, OC, OH, OW), device='cuda')
    
    # Weight needs to be transposed to match kernel's expected layout (OC, IC, K, K)
    w_t = conv_transpose_weight.transpose(0, 1).contiguous()
    
    fused_ext.fused_op(
        x, 
        w_t, 
        bias.view(-1), 
        out,
        stride_h,
        pad_h
    )
    
    return out

# Placeholder parameters as defined in the prompt
batch_size = 32
in_channels  = 64  
out_channels = 64  
height = width = 256 
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
