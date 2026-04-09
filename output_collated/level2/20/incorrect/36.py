# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_132207/code_6.py
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

# Optimized CUDA kernel implementing Implicit GEMM for ConvTranspose3D with fused post-processing
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16

__global__ void fused_conv_transpose_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int D, int H, int W,
    int OC, int KD, int KH, int KW,
    int SD, int SH, int SW,
    int PD, int PH, int PW,
    int OD, int OH, int OW
) {
    // Shared memory for tiles
    __shared__ float shmem_input[TILE_M][TILE_K + 1];  // +1 to avoid bank conflicts
    __shared__ float shmem_weight[TILE_K][TILE_N + 1];
    
    // Thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    
    // Calculate output indices
    int od = bz / (OH * OW);
    int oh = (bz % (OH * OW)) / OW;
    int ow = (bz % (OH * OW)) % OW;
    
    // Register arrays for accumulation
    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Loop over input channels in tiles
    for (int k = 0; k < C; k += TILE_K) {
        // Cooperatively load input tile
        for (int i = 0; i < TILE_M; i += blockDim.y) {
            for (int j = 0; j < TILE_K; j += blockDim.x) {
                int idx = i + j * TILE_M;
                if (tx + j < TILE_K && ty + i < TILE_M) {
                    int c = k + tx + j;
                    int n = by * TILE_M + ty + i;
                    if (c < C && n < N) {
                        shmem_input[ty + i][tx + j] = input[n * C * D * H * W + c * D * H * W];
                    } else {
                        shmem_input[ty + i][tx + j] = 0.0f;
                    }
                }
            }
        }
        
        // Cooperatively load weight tile
        for (int i = 0; i < TILE_K; i += blockDim.y) {
            for (int j = 0; j < TILE_N; j += blockDim.x) {
                int idx = i + j * TILE_K;
                if (tx + j < TILE_N && ty + i < TILE_K) {
                    int c = k + ty + i;
                    int oc = bx * TILE_N + tx + j;
                    if (c < C && oc < OC) {
                        // Compute weight index based on transposed convolution mapping
                        int kd_base = (od + PD) % SD;
                        int kh_base = (oh + PH) % SH;
                        int kw_base = (ow + PW) % SW;
                        
                        int kd = kd_base == 0 ? (od + PD) / SD : -1;
                        int kh = kh_base == 0 ? (oh + PH) / SH : -1;
                        int kw = kw_base == 0 ? (ow + PW) / SW : -1;
                        
                        if (kd >= 0 && kd < KD && kh >= 0 && kh < KH && kw >= 0 && kw < KW) {
                            int flip_kd = KD - 1 - kd;
                            int flip_kh = KH - 1 - kh;
                            int flip_kw = KW - 1 - kw;
                            shmem_weight[ty + i][tx + j] = weight[oc * C * KD * KH * KW + 
                                                                  c * KD * KH * KW + 
                                                                  flip_kd * KH * KW + 
                                                                  flip_kh * KW + 
                                                                  flip_kw];
                        } else {
                            shmem_weight[ty + i][tx + j] = 0.0f;
                        }
                    } else {
                        shmem_weight[ty + i][tx + j] = 0.0f;
                    }
                }
            }
        }
        
        __syncthreads();
        
        // Compute partial products
        for (int i = 0; i < 4; i++) {
            for (int kk = 0; kk < TILE_K; kk++) {
                accum[i] += shmem_input[ty * 4 + i][kk] * shmem_weight[kk][tx];
            }
        }
        
        __syncthreads();
    }
    
    // Write results with fused post-processing
    for (int i = 0; i < 4; i++) {
        int n = by * TILE_M + ty * 4 + i;
        int oc = bx * TILE_N + tx;
        int idx = n * OC * OD * OH * OW + oc * OD * OH * OW + od * OH * OW + oh * OW + ow;
        
        if (n < N && oc < OC) {
            float val = accum[i];
            // Fused post-processing: ((x + bias) + x) * x + x
            val = ((val + bias[oc]) + val) * val + val;
            output[idx] = val;
        }
    }
}

void fused_conv_transpose_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w
) {
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    
    int OC = weight.size(0);
    int KD = weight.size(2);
    int KH = weight.size(3);
    int KW = weight.size(4);
    
    // Calculate output dimensions
    int OD = (D - 1) * stride_d - 2 * padding_d + KD;
    int OH = (H - 1) * stride_h - 2 * padding_h + KH;
    int OW = (W - 1) * stride_w - 2 * padding_w + KW;
    
    // Grid and block dimensions
    dim3 block_dim(16, 16);
    dim3 grid_dim((OC + TILE_N - 1) / TILE_N, (N + TILE_M - 1) / TILE_M, OD * OH * OW);
    
    fused_conv_transpose_bias_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        OC, KD, KH, KW,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        OD, OH, OW
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w
);

torch::Tensor fused_conv_transpose(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w
) {
    // Calculate output dimensions
    int N = input.size(0);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    int OC = weight.size(0);
    int KD = weight.size(2);
    int KH = weight.size(3);
    int KW = weight.size(4);
    
    int OD = (D - 1) * stride_d - 2 * padding_d + KD;
    int OH = (H - 1) * stride_h - 2 * padding_h + KH;
    int OW = (W - 1) * stride_w - 2 * padding_w + KW;
    
    auto output = torch::empty({N, OC, OD, OH, OW}, 
                               torch::dtype(input.dtype()).device(input.device()));
    
    fused_conv_transpose_forward(input, weight, bias, output,
                                 stride_d, stride_h, stride_w,
                                 padding_d, padding_h, padding_w);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose", &fused_conv_transpose, "Fused ConvTranspose3D with bias");
}
"""

fused_ext = load_inline(
    name='fused_conv_transpose_ext',
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
    # Verify groups is 1 (our kernel doesn't support grouped convolutions)
    if conv_transpose_groups != 1:
        raise ValueError("Grouped convolutions not supported in this implementation")
    
    # Verify dilation is (1,1,1) (our kernel doesn't support dilation)
    if conv_transpose_dilation != (1, 1, 1):
        raise ValueError("Dilation not supported in this implementation")
    
    # Call our optimized fused kernel
    result = fused_ext.fused_conv_transpose(
        x,
        conv_transpose_weight,
        bias.view(-1),
        conv_transpose_stride[0], conv_transpose_stride[1], conv_transpose_stride[2],
        conv_transpose_padding[0], conv_transpose_padding[1], conv_transpose_padding[2]
    )
    
    # Add output padding if needed
    if conv_transpose_output_padding != (0, 0, 0):
        pad_d, pad_h, pad_w = conv_transpose_output_padding
        result = torch.nn.functional.pad(result, (0, pad_w, 0, pad_h, 0, pad_d))
    
    return result

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
