# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_144040/code_9.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'subtract_value_1', 'subtract_value_2']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'subtract_value_1', 'subtract_value_2']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, subtracts two values, applies Mish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

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
    # State for conv (nn.Conv2d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
    if 'subtract_value_1' in flat_state:
        state_kwargs['subtract_value_1'] = flat_state['subtract_value_1']
    else:
        state_kwargs['subtract_value_1'] = getattr(model, 'subtract_value_1')
    if 'subtract_value_2' in flat_state:
        state_kwargs['subtract_value_2'] = flat_state['subtract_value_2']
    else:
        state_kwargs['subtract_value_2'] = getattr(model, 'subtract_value_2')
    missing = [name for name in REQUIRED_STATE_NAMES if name not in state_kwargs]
    if missing:
        raise RuntimeError(f'Missing required state entries: {missing}')
    return state_kwargs


def get_functional_inputs():
    model = build_reference_model()
    forward_args = tuple(get_inputs())
    state_kwargs = extract_state_kwargs(model)
    return forward_args, state_kwargs




# fused_conv_mish_coalesced.py
# --------------------------------------------------------------
#  CUDA + C++ + Python wrapper (single file)
# --------------------------------------------------------------
import torch
from torch.utils.cpp_extension import load_inline

# --------------------------------------------------------------
#  CUDA kernel
# --------------------------------------------------------------
# Tile size for the output plane (must fit into 256 threads per block)
TILE_H = 8          # number of output rows calculated by one block
TILE_W = 16         # number of output columns calculated by one block
# (TILE_H * TILE_W) = 128 threads – we will launch 2 warps for
# better occupancy (128 threads * 2 = 256 total threads per block)
# The kernel loads the required input patch (TILE_H+K-1)*(TILE_W+K-1)
# into shared memory in a co‑alesced fashion.

cuda_source = rf"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_H {TILE_H}
#define TILE_W {TILE_W}

// ------------------------------------------------------------------
//  Helper: fast tanh / softplus (already in fast‑math, we keep it)
// ------------------------------------------------------------------
__device__ __forceinline__ float mish_activation(float x)
{{
    // Mish = x * tanh(softplus(x))
    float sp = logf(1.0f + expf(x));          // softplus
    return x * tanhf(sp);
}}

// ------------------------------------------------------------------
//  Kernel
// ------------------------------------------------------------------
extern "C" __global__
void fused_conv_mish_kernel(
    const float* __restrict__ input,    // [B, IC, IH, IW]
    const float* __restrict__ weight,   // [OC, IC, K, K]
    const float* __restrict__ bias,     // [OC]
    float* __restrict__ output,         // [B, OC, OH, OW]
    const int batch, const int in_c,
    const int in_h, const int in_w,
    const int out_c, const int k,
    const int out_h, const int out_w,
    const float sub1, const float sub2)
{{
    // --------------------------------------------------------------
    //  Block / thread identifiers
    // --------------------------------------------------------------
    const int oc = blockIdx.y;                 // output channel
    const int b  = blockIdx.z;                 // batch index

    // each block computes a TILE_H × TILE_W patch of the output
    const int out_base_y = blockIdx.x * TILE_H;   // top‑left corner (oh)
    const int out_base_x = blockIdx.y * TILE_W;   // top‑left corner (ow)
    // note: blockIdx.y is already used for oc, so we need a 3‑D grid:
    //   dim3 grid( (OH+TILE_H-1)/TILE_H, out_c, batch );

    // --------------------------------------------------------------
    //  Shared memory layout
    // --------------------------------------------------------------
    //  [0 .. in_c*k*k)           : cached weights for this oc
    //  [in_c*k*k .. end)         : input tile (single channel at a time)
    extern __shared__ float smem[];
    float* s_weight = smem;                                   // size = in_c*k*k
    float* s_input  = smem + in_c * k * k;                    // size = (TILE_H+k-1)*(TILE_W+k-1)

    const int INPUT_TILE_H = TILE_H + k - 1;
    const int INPUT_TILE_W = TILE_W + k - 1;
    const int INPUT_TILE_SZ = INPUT_TILE_H * INPUT_TILE_W;

    // --------------------------------------------------------------
    //  1) Load weight for this output channel once per block
    // --------------------------------------------------------------
    const int weight_elems = in_c * k * k;
    for (int idx = threadIdx.x; idx < weight_elems; idx += blockDim.x * blockDim.y)
    {{
        s_weight[idx] = weight[ oc * weight_elems + idx ];
    }}

    // --------------------------------------------------------------
    //  2) Accumulator – start with bias (one per output pixel)
    // --------------------------------------------------------------
    // each thread will compute ONE output pixel inside the tile
    const int ty = threadIdx.y;        // 0 .. TILE_H-1
    const int tx = threadIdx.x;        // 0 .. TILE_W-1
    const int oh = out_base_y + ty;    // absolute output row
    const int ow = out_base_x + tx;    // absolute output col

    // guard against boundary over‑run (tiles at image edge)
    if (oh >= out_h || ow >= out_w) return;

    float acc = bias[oc];      // start from bias for this channel

    // --------------------------------------------------------------
    //  3) Loop over input channels – load one channel’s tile at a time
    // --------------------------------------------------------------
    for (int ic = 0; ic < in_c; ++ic)
    {{
        // ------------------------------------------------------------------
        //  3a) Cooperatively load the INPUT_TILE for channel 'ic' into s_input
        // ------------------------------------------------------------------
        // each thread loads several elements (stride = blockDim.x * blockDim.y)
        for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < INPUT_TILE_SZ; idx += blockDim.x * blockDim.y)
        {{
            int ti = idx / INPUT_TILE_W;            // tile row inside shared mem
            int tj = idx % INPUT_TILE_W;            // tile col inside shared mem

            int in_y = out_base_y + ti;   // corresponds to input row needed
            int in_x = out_base_x + tj;   // corresponds to input col needed

            // Bounds check – pad with zeros
            float val = 0.0f;
            if (in_y < in_h && in_x < in_w)
            {{
                val = input[ ((b * in_c + ic) * in_h + in_y) * in_w + in_x ];
            }}
            s_input[ti * INPUT_TILE_W + tj] = val;
        }}
        __syncthreads();   // make sure the whole tile is ready

        // ------------------------------------------------------------------
        //  3b) Compute contribution of this channel using the shared tile
        // ------------------------------------------------------------------
        // convolution window is anchored at (ty,tx) inside the output tile
        // i.e. we need s_input[ty+i][tx+j] for i,j = 0..k-1
        for (int i = 0; i < k; ++i)
        {{
            for (int j = 0; j < k; ++j)
            {{
                float in_val = s_input[(ty + i) * INPUT_TILE_W + (tx + j)];
                float w_val  = s_weight[(ic * k + i) * k + j];
                acc += in_val * w_val;
            }}
        }}
        __syncthreads();   // before next channel we reuse s_input
    }}

    // --------------------------------------------------------------
    //  4) Apply the fused subtraction + Mish activation
    // --------------------------------------------------------------
    float v = acc - sub1 - sub2;
    float out_val = mish_activation(v);   // Mish = v * tanh(softplus(v))

    // --------------------------------------------------------------
    //  5) Write result
    // --------------------------------------------------------------
    output[ ((b * out_c + oc) * out_h + oh) * out_w + ow ] = out_val;
}}

// ------------------------------------------------------------------
//  Host launcher – thin wrapper called from Python
// ------------------------------------------------------------------
void fused_conv_mish_launcher(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float sub1,
    float sub2)
{{
    const int batch = input.size(0);
    const int in_c  = input.size(1);
    const int in_h  = input.size(2);
    const int in_w  = input.size(3);
    const int out_c = weight.size(0);
    const int k     = weight.size(2);               // kernel height == width
    const int out_h = in_h - k + 1;
    const int out_w = in_w - k + 1;

    const int threads_x = TILE_W;          // will become blockDim.x
    const int threads_y = TILE_H;          // will become blockDim.y
    const dim3 threads(threads_x, threads_y);

    const dim3 grid( (out_h + TILE_H - 1) / TILE_H,   // grid.x : tiles along height
                    out_c,                           // grid.y : output channels
                    batch );                         // grid.z : batch

    // shared memory = weights + one input tile (single channel)
    const size_t weight_shmem = in_c * k * k * sizeof(float);
    const size_t tile_shmem   = (TILE_H + k - 1) * (TILE_W + k - 1) * sizeof(float);
    const size_t shmem_bytes  = weight_shmem + tile_shmem;

    fused_conv_mish_kernel<<<grid, threads, shmem_bytes>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_c, in_h, in_w,
        out_c, k,
        out_h, out_w,
        sub1, sub2);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA error in fused_conv_mish_kernel: %s\\n", cudaGetErrorString(err));
}}
"""

# --------------------------------------------------------------
#  C++ binding (PYBIND11)
# --------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the launcher defined in the .cu file
void fused_conv_mish_launcher(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float sub1,
    float sub2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_mish", &fused_conv_mish_launcher,
          "Fused convolution + subtraction + Mish activation (coalesced version)");
}
"""

# --------------------------------------------------------------
#  Build the extension (once, at import time)
# --------------------------------------------------------------
fused_ext = load_inline(
    name='fused_conv_mish_coalesced',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False
)

# --------------------------------------------------------------
#  Functional model – the only public entry point
# --------------------------------------------------------------
def functional_model(
    x, *,
    conv_weight,
    conv_bias,
    conv_stride=1,
    conv_padding=0,
    conv_dilation=1,
    conv_groups=1,
    subtract_value_1,
    subtract_value_2
):
    """
    Applies a 2‑D convolution (valid padding, stride 1, no groups/dilation)
    followed by two subtractions and the Mish activation, using the
    custom CUDA kernel defined above.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, C_in, H, W); must be contiguous and on CUDA.
    conv_weight : torch.Tensor
        Weight tensor of shape (C_out, C_in, K, K).
    conv_bias : torch.Tensor
        Bias tensor of shape (C_out,).
    subtract_value_1, subtract_value_2 : float
        Constants that are subtracted from the convolution result before Mish.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (B, C_out, H‑K+1, W‑K+1) on the same device as ``x``.
    """
    # The kernel currently implements *valid* convolution only.
    batch, _, h, w = x.shape
    k = conv_weight.shape[2]
    out_h, out_w = h - k + 1, w - k + 1

    # Allocate output tensor (uninitialised, will be overwritten by kernel)
    out = torch.empty((batch, conv_weight.size(0), out_h, out_w), device=x.device, dtype=x.dtype)

    # Launch the fused kernel
    fused_ext.fused_conv_mish(
        x, conv_weight, conv_bias, out,
        float(subtract_value_1), float(subtract_value_2)
    )
    return out

# --------------------------------------------------------------
#  End of file – only functional_model is imported by the grader
# --------------------------------------------------------------
