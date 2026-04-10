# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_140117/code_6.py
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

# Optimized fused kernel using tiled convolution with shared memory and fused post-processing
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_DIM 32
#define CHANNEL_TILE 32

__global__ void fused_conv_transpose3d_post_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ post_bias,
    float* __restrict__ output,
    int N, int IC, int OC,
    int ID, int IH, int IW,
    int KD, int KH, int KW,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    // Shared memory for weight and input tiles
    extern __shared__ float shared_mem[];
    float* shared_weight = shared_mem;
    float* shared_input = shared_mem + CHANNEL_TILE * KD * KH * KW;
    
    int od = blockIdx.z;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    
    int OD = (ID - 1) * stride_d - 2 * padding_d + dilation_d * (KD - 1) + output_padding_d + 1;
    int OH = (IH - 1) * stride_h - 2 * padding_h + dilation_h * (KH - 1) + output_padding_h + 1;
    int OW = (IW - 1) * stride_w - 2 * padding_w + dilation_w * (KW - 1) + output_padding_w + 1;
    
    if (od >= OD || oh >= OH || ow >= OW) return;
    
    int oc_start = blockIdx.z % OC;
    
    for (int oc_base = 0; oc_base < OC; oc_base += CHANNEL_TILE) {
        int oc_end = min(oc_base + CHANNEL_TILE, OC);
        int oc_tile_size = oc_end - oc_base;
        
        // Load weight tile to shared memory
        for (int i = threadIdx.y * blockDim.x + threadIdx.x; 
             i < oc_tile_size * KD * KH * KW; 
             i += blockDim.y * blockDim.x) {
            int local_oc = i / (KD * KH * KW);
            int kd = (i % (KD * KH * KW)) / (KH * KW);
            int kh = (i % (KH * KW)) / KW;
            int kw = i % KW;
            shared_weight[i] = weight[((oc_base + local_oc) * IC + threadIdx.z) * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw];
        }
        
        __syncthreads();
        
        float result = 0.0f;
        
        // Perform convolution computation
        for (int ic = 0; ic < IC; ++ic) {
            for (int kd = 0; kd < KD; ++kd) {
                for (int kh = 0; kh < KH; ++kh) {
                    for (int kw = 0; kw < KW; ++kw) {
                        int id = od + padding_d - kd * dilation_d;
                        int ih = oh + padding_h - kh * dilation_h;
                        int iw = ow + padding_w - kw * dilation_w;
                        
                        if (id % stride_d == 0 && ih % stride_h == 0 && iw % stride_w == 0) {
                            id /= stride_d;
                            ih /= stride_h;
                            iw /= stride_w;
                            
                            if (id >= 0 && id < ID && ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                float val = input[((blockIdx.z % N) * IC + ic) * (ID * IH * IW) + id * (IH * IW) + ih * IW + iw];
                                result += val * shared_weight[((oc_base % CHANNEL_TILE) * KD + kd) * (KH * KW) + kh * KW + kw];
                            }
                        }
                    }
                }
            }
        }
        
        // Add bias and apply post-processing
        float b = conv_bias[oc_base + (oc_base % CHANNEL_TILE)];
        float x = result + b;
        float post_b = post_bias[oc_base + (oc_base % CHANNEL_TILE)];
        float final_result = ((x + post_b) + x) * x + x;
        
        // Write output
        if (oc_base == 0) { // Only write once per output channel
            output[((blockIdx.z % N) * OC + (oc_base + (oc_base % CHANNEL_TILE))) * (OD * OH * OW) + od * (OH * OW) + oh * OW + ow] = final_result;
        }
        
        __syncthreads();
    }
}

void fused_conv_transpose3d_post_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    int N = input.size(0);
    int IC = input.size(1);
    int ID = input.size(2);
    int IH = input.size(3);
    int IW = input.size(4);
    
    int OC = weight.size(1);
    int KD = weight.size(2);
    int KH = weight.size(3);
    int KW = weight.size(4);
    
    int OD = (ID - 1) * stride_d - 2 * padding_d + dilation_d * (KD - 1) + output_padding_d + 1;
    int OH = (IH - 1) * stride_h - 2 * padding_h + dilation_h * (KH - 1) + output_padding_h + 1;
    int OW = (IW - 1) * stride_w - 2 * padding_w + dilation_w * (KW - 1) + output_padding_w + 1;
    
    dim3 block(16, 16, 1);
    dim3 grid((OW + block.x - 1) / block.x, 
              (OH + block.y - 1) / block.y, 
              N * OC);
    
    size_t shared_mem_size = CHANNEL_TILE * KD * KH * KW * sizeof(float) + 
                            block.x * block.y * sizeof(float);
    
    fused_conv_transpose3d_post_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        post_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, IC, OC, ID, IH, IW,
        KD, KH, KW,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_post_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w
);

torch::Tensor fused_conv_transpose3d_post(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t padding_d, int64_t padding_h, int64_t padding_w,
    int64_t output_padding_d, int64_t output_padding_h, int64_t output_padding_w,
    int64_t dilation_d, int64_t dilation_h, int64_t dilation_w
) {
    // Calculate output dimensions
    int64_t N = input.size(0);
    int64_t IC = input.size(1);
    int64_t ID = input.size(2);
    int64_t IH = input.size(3);
    int64_t IW = input.size(4);
    
    int64_t OC = weight.size(1);
    int64_t KD = weight.size(2);
    int64_t KH = weight.size(3);
    int64_t KW = weight.size(4);
    
    int64_t OD = (ID - 1) * stride_d - 2 * padding_d + dilation_d * (KD - 1) + output_padding_d + 1;
    int64_t OH = (IH - 1) * stride_h - 2 * padding_h + dilation_h * (KH - 1) + output_padding_h + 1;
    int64_t OW = (IW - 1) * stride_w - 2 * padding_w + dilation_w * (KW - 1) + output_padding_w + 1;
    
    auto output = torch::empty({N, OC, OD, OH, OW}, input.options());
    fused_conv_transpose3d_post_forward(
        input, weight, conv_bias, post_bias, output,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_post", &fused_conv_transpose3d_post, "Fused ConvTranspose3d with post-processing");
}
"""

fused_ext = load_inline(
    name='fused_conv_transpose3d_post_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
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
    # Assuming groups=1 for simplicity
    if conv_transpose_groups != 1:
        raise NotImplementedError("Grouped convolutions not implemented in this optimized version")
    
    # Extract stride, padding, output_padding, dilation for each dimension
    stride_d, stride_h, stride_w = conv_transpose_stride
    padding_d, padding_h, padding_w = conv_transpose_padding
    output_padding_d, output_padding_h, output_padding_w = conv_transpose_output_padding
    dilation_d, dilation_h, dilation_w = conv_transpose_dilation
    
    # Flatten bias for simplified kernel indexing
    bias_flat = bias.view(-1)
    
    # Use optimized fused kernel for the convolution and post-processing
    return fused_ext.fused_conv_transpose3d_post(
        x, conv_transpose_weight, conv_transpose_bias, bias_flat,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w
    )

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
