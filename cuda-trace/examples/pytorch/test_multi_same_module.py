import torch
from torch import nn
from torch.utils.cpp_extension import load_inline


cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename T>
__global__ void relu_div_kernel(const T* inp, T* out, const T div, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T val = inp[idx];
        val = (val > static_cast<T>(0)) ? val : static_cast<T>(0);
        out[idx] = val / div;
    }
}

template<typename T>
__global__ void bias_add_kernel(const T* inp, T* out, const T bias, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = inp[idx] + bias;
    }
}

torch::Tensor fused_two_kernels_cuda(torch::Tensor input, double divisor, double bias) {
    auto tmp = torch::empty_like(input);
    auto output = torch::empty_like(input);
    const size_t size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fused_two_kernels", [&] {
        relu_div_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            tmp.data_ptr<scalar_t>(),
            static_cast<scalar_t>(divisor),
            size
        );
        bias_add_kernel<scalar_t><<<blocks, threads>>>(
            tmp.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            static_cast<scalar_t>(bias),
            size
        );
    });
    cudaDeviceSynchronize();
    return output;
}
"""


cpp_source = r"""
#include <torch/extension.h>

torch::Tensor fused_two_kernels_cuda(torch::Tensor input, double divisor, double bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_two_kernels_cuda, "Two-kernel CUDA operation");
}
"""


mod = load_inline(
    name="fused_multi_same_module",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)


class ModelNew(nn.Module):
    def __init__(self, in_dim, out_dim, divisor, bias):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.divisor = float(divisor)
        self.bias = float(bias)

    def forward(self, x):
        return mod.forward(self.linear(x), self.divisor, self.bias)


def main():
    batch_size = 256
    in_dim = 1024
    out_dim = 2048
    divisor = 5.0
    bias = 1.25

    model = ModelNew(in_dim, out_dim, divisor, bias).cuda()
    input_tensor = torch.rand(batch_size, in_dim).cuda()
    output = model(input_tensor)
    print(output[:4, :8])


main()
