import torch
from torch import nn
from torch.utils.cpp_extension import load_inline

# --- CUDA Kernel Implementation ---
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename T>
__global__ void fused_kernel(const T* inp, T* out, const T div, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        printf("Test\n");
    }
    if (idx < n) {
        T val = inp[idx];
        val = (val > static_cast<T>(0))? val : static_cast<T>(0);
        out[idx] = val / div;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor input, double divisor) {
    auto output = torch::empty_like(input);
    const size_t size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fused_kernel", [&] {
        fused_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            static_cast<scalar_t>(divisor),
            size
        );
    });
    cudaDeviceSynchronize();
    return output;
}
"""

# --- C++ Bindings ---
cpp_source = r"""
#include <torch/extension.h>

torch::Tensor fused_op_cuda(torch::Tensor input, double divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_op_cuda, "Fused ReLU+Div operation");
}
"""

# Compile and load extension
fused_mod = load_inline(
    name="fused_activation",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True
)

# --- Optimized Model ---
class ModelNew(nn.Module):
    def __init__(self, in_dim, out_dim, divisor):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.divisor = float(divisor)
    
    def forward(self, x):
        lin_out = self.linear(x)
        return fused_mod.forward(lin_out, self.divisor)

def main():
    batch_size = 1024
    in_dim = 8192
    out_dim = 8192
    divisor = 10.0

    model = ModelNew(in_dim, out_dim, divisor).cuda()
    input_tensor = torch.rand(batch_size, in_dim).cuda()
    
    output = model(input_tensor)
    print(output[:10])

main()
