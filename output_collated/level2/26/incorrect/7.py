# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_040909/code_3.py
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

# CUDA kernel source code with tiled tensor core-like shared memory computation
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_DIM 8
#define THREAD_DIM_X 8
#define THREAD_DIM_Y 8
#define THREAD_DIM_Z 4

__global__ void fused_conv_transpose3d_hardswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    int B, int IC, int OC,
    int ID, int IH, int IW,
    int OD, int OH, int OW,
    int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW) {

    // Shared memory for caching input and weight tiles
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + TILE_DIM * TILE_DIM * TILE_DIM * TILE_DIM;

    int oc = blockIdx.x * blockDim.x + threadIdx.x; // Output channel
    int batch_idx = blockIdx.y; // Batch index
    int od = blockIdx.z * blockDim.z + threadIdx.z; // Output depth
    int oh = threadIdx.y; // Output height (limited by block size)
    int ow = threadIdx.x; // Output width (limited by block size)

    // Each thread computes one output element
    if (oc < OC && od < OD && batch_idx < B) {
        float val = (oc < OC) ? bias[oc] : 0.0f;

        // Perform convolution calculation
        if (oc < OC) {
            for (int ic = 0; ic < IC; ++ic) {
                for (int kd = 0; kd < kD; ++kd) {
                    for (int kh = 0; kh < kH; ++kh) {
                        for (int kw = 0; kw < kW; ++kw) {
                            // Calculate corresponding input position
                            int id = (od + padD - kd) / strideD;
                            int ih = (oh + padH - kh) / strideH;
                            int iw = (ow + padW - kw) / strideW;

                            // Check bounds and stride divisibility
                            if ((od + padD - kd) % strideD == 0 &&
                                (oh + padH - kh) % strideH == 0 &&
                                (ow + padW - kw) % strideW == 0 &&
                                id >= 0 && id < ID &&
                                ih >= 0 && ih < IH &&
                                iw >= 0 && iw < IW) {
                                val += input[batch_idx * IC * ID * IH * IW +
                                             ic * ID * IH * IW +
                                             id * IH * IW +
                                             ih * IW +
                                             iw] *
                                       weight[oc * IC * kD * kH * kW +
                                              ic * kD * kH * kW +
                                              kd * kH * kW +
                                              kh * kW +
                                              kw];
                            }
                        }
                    }
                }
            }
        }

        // Add add_input tensor value and apply HardSwish activation
        if (oc < OC && od < OD) {
            float add_val = add_input[batch_idx * OC * OD * OH * OW +
                                      oc * OD * OH * OW +
                                      od * OH * OW +
                                      oh * OW +
                                      ow];
            val += add_val;
            
            // HardSwish: x * relu6(x + 3) / 6
            float h_val = val * fminf(fmaxf(val + 3.0f, 0.0f), 6.0f) / 6.0f;
            output[batch_idx * OC * OD * OH * OW +
                   oc * OD * OH * OW +
                   od * OH * OW +
                   oh * OW +
                   ow] = h_val;
        }
    }
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor add_input,
    torch::Tensor output,
    int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW) {
    
    int B = input.size(0);
    int IC = input.size(1);
    int ID = input.size(2);
    int IH = input.size(3);
    int IW = input.size(4);
    
    int OC = weight.size(0);
    int OD = output.size(2);
    int OH = output.size(3);
    int OW = output.size(4);

    // Grid and block dimensions
    dim3 grid(OC / THREAD_DIM_X + 1, B, OD / THREAD_DIM_Z + 1);
    dim3 block(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z);

    // Shared memory size estimation (could be optimized further)
    size_t shared_mem_size = (TILE_DIM * TILE_DIM * TILE_DIM * TILE_DIM * 2) * sizeof(float);

    fused_conv_transpose3d_hardswish_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        kD, kH, kW,
        strideD, strideH, strideW,
        padD, padH, padW
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor add_input,
    torch::Tensor output,
    int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused ConvTranspose3d + Bias + HardSwish");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
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
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    bias,
):
    # Move inputs to GPU if not already there
    x = x.cuda()
    add_input = add_input.cuda()
    conv_transpose_weight = conv_transpose_weight.cuda()
    conv_transpose_bias = conv_transpose_bias.cuda()

    # Get dimensions
    B, IC, ID, IH, IW = x.shape
    OC = conv_transpose_weight.shape[0]
    kD, kH, kW = conv_transpose_weight.shape[2], conv_transpose_weight.shape[3], conv_transpose_weight.shape[4]
    strideD, strideH, strideW = conv_transpose_stride
    padD, padH, padW = conv_transpose_padding
    out_D = (ID - 1) * strideD - 2 * padD + kD + conv_transpose_output_padding[0]
    out_H = (IH - 1) * strideH - 2 * padH + kH + conv_transpose_output_padding[1]
    out_W = (IW - 1) * strideW - 2 * padW + kW + conv_transpose_output_padding[2]

    # Create output tensor
    output = torch.empty((B, OC, out_D, out_H, out_W), dtype=x.dtype, device=x.device)

    # Call the fused operation
    fused_ext.fused_op(
        x.contiguous(), 
        conv_transpose_weight.contiguous(), 
        conv_transpose_bias.contiguous(), 
        add_input.contiguous(), 
        output,
        kD, kH, kW,
        strideD, strideH, strideW,
        padD, padH, padW
    )

    return output
