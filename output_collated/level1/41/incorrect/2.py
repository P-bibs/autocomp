# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_035026/code_18.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['kernel_size', 'stride', 'padding', 'dilation', 'return_indices']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['maxpool_kernel_size', 'maxpool_stride', 'maxpool_padding', 'maxpool_dilation', 'maxpool_ceil_mode', 'maxpool_return_indices']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 1D.
    """

    def __init__(self, kernel_size: int, stride: int=None, padding: int=0, dilation: int=1, return_indices: bool=False):
        """
        Initializes the Max Pooling 1D layer.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
            padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices)

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
    # State for maxpool (nn.MaxPool1d)
    state_kwargs['maxpool_kernel_size'] = model.maxpool.kernel_size
    state_kwargs['maxpool_stride'] = model.maxpool.stride
    state_kwargs['maxpool_padding'] = model.maxpool.padding
    state_kwargs['maxpool_dilation'] = model.maxpool.dilation
    state_kwargs['maxpool_ceil_mode'] = model.maxpool.ceil_mode
    state_kwargs['maxpool_return_indices'] = model.maxpool.return_indices
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

# Global variable to store compiled extension
fused_ext = None

def functional_model(
    x,
    *,
    maxpool_kernel_size,
    maxpool_stride,
    maxpool_padding,
    maxpool_dilation,
    maxpool_ceil_mode,
    maxpool_return_indices,
):
    global fused_ext
    
    if fused_ext is None:
        cuda_source = r"""
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void maxpool1d_shared_kernel(
            const float* __restrict__ input,
            float* __restrict__ output,
            int batch_size,
            int channels,
            int input_len,
            int output_len,
            int k_size,
            int stride,
            int pad,
            int dil
        ) {
            // Each block handles one (batch, channel)
            int bc = blockIdx.x;
            int b = bc / channels;
            int c = bc % channels;
            
            const float* input_row = input + bc * input_len;
            float* output_row = output + bc * output_len;
            
            // Shared memory buffer
            extern __shared__ float s_data[];
            
            // Thread-cooperative loading of the whole row into shared memory
            // This is efficient because threads access global memory contiguously
            for (int i = threadIdx.x; i < input_len; i += blockDim.x) {
                s_data[i] = input_row[i];
            }
            __syncthreads();
            
            for (int i = threadIdx.x; i < output_len; i += blockDim.x) {
                int start = i * stride - pad;
                float max_val = -1e38; // Sufficient for float
                
                for (int k = 0; k < k_size; ++k) {
                    int pos = start + k * dil;
                    if (pos >= 0 && pos < input_len) {
                        float val = s_data[pos];
                        if (val > max_val) max_val = val;
                    }
                }
                output_row[i] = max_val;
            }
        }

        void maxpool1d_optimized(
            const torch::Tensor& input,
            torch::Tensor& output,
            int k_s, int stride, int pad, int dil
        ) {
            int batch = input.size(0);
            int chans = input.size(1);
            int in_l = input.size(2);
            int out_l = output.size(2);
            
            int threads = 256;
            int blocks = batch * chans;
            
            // Shared memory size is input_len * sizeof(float)
            size_t shared_size = in_l * sizeof(float);
            
            maxpool1d_shared_kernel<<<blocks, threads, shared_size>>>(
                input.data_ptr<float>(), output.data_ptr<float>(),
                batch, chans, in_l, out_l, k_s, stride, pad, dil
            );
        }
        """
        
        cpp_source = r"""
        void maxpool1d_optimized(const torch::Tensor& input, torch::Tensor& output, int k_s, int stride, int pad, int dil);
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("maxpool1d", &maxpool1d_optimized, "Optimized MaxPool1D");
        }
        """
        
        fused_ext = load_inline(
            name='fused_maxpool1d_opt',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            with_cuda=True
        )

    # Ensure input is on GPU and float32
    x = x.to(device='cuda', dtype=torch.float32)
    
    # Calculate output dimensions
    if maxpool_ceil_mode:
        output_length = (x.size(2) + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1 + maxpool_stride - 1) // maxpool_stride + 1
    else:
        output_length = (x.size(2) + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
    
    output = torch.empty((x.size(0), x.size(1), output_length), device='cuda', dtype=x.dtype)
    
    fused_ext.maxpool1d(x, output, int(maxpool_kernel_size), int(maxpool_stride), int(maxpool_padding), int(maxpool_dilation))
    
    return output
