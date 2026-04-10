# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_193747/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a ReLU activation.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

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

# Optimized CUDA kernel using vectorized loads and avoiding grid re-calculation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_forward_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx = tid * 4;
    
    if (idx + 3 < n) {
        float4 in_vec = reinterpret_cast<const float4*>(input)[tid];
        float4 out_vec;
        out_vec.x = (in_vec.x > 0.0f) ? in_vec.x : 0.0f;
        out_vec.y = (in_vec.y > 0.0f) ? in_vec.y : 0.0f;
        out_vec.z = (in_vec.z > 0.0f) ? in_vec.z : 0.0f;
        out_vec.w = (in_vec.w > 0.0f) ? in_vec.w : 0.0f;
        reinterpret_cast<float4*>(output)[tid] = out_vec;
    } else {
        for (int i = 0; i < 4; ++i) {
            if (idx + i < n) {
                output[idx + i] = (input[idx + i] > 0.0f) ? input[idx + i] : 0.0f;
            }
        }
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    size_t num_elements = input.numel();
    size_t vec_elements = (num_elements + 3) / 4;  // Ceiling division
    const int threads = 256;
    const int blocks = (vec_elements + threads - 1) / threads;
    
    fused_op_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        num_elements
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Optimized ReLU with vectorized memory access");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Use CUDA Graphs for execution
class FusedOpExecutor:
    def __init__(self):
        self.graph = None
        self.static_input = None
        self.static_output = None
        
    def execute(self, x):
        # Lazy initialization of CUDA Graph for the given input size
        if self.graph is None or self.static_input.shape != x.shape:
            self.static_input = torch.empty_like(x)
            self.static_output = torch.empty_like(x)
            
            # Capture the graph
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                g = torch.cuda.CUDAGraph()
                with g:
                    fused_ext.fused_op(self.static_input, self.static_output)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = g
            
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output.clone()

executor = FusedOpExecutor()

def functional_model(x):
    return executor.execute(x)

# --- Evaluation setup ---
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]
