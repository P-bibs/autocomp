Produce a very straightforward implementation of this pytorch model in CUDA. Do not optimize. Do not fuse. The implementation should be as simple as possible (like a beginner would write) and easy to verify correctness of. You may assume that the provided constants and tensor shapes are the only shapes your code needs to work with. It does not need to be general.

Your response should consist of two things: a C code block and a json code block. If you are confused or have questions about the task, they can come at the end.

The C code block should contain a series of CUDA kernels.

The json code block should be a specification of how to call those kernels such that they produce the desired output. It should be in a very specific format which matches this json example:

```json
{
  "source_tensors": {
    "a": { "size": 16 },
    "b": { "size": 16 },
    "x": { "size": 16 }
  },
  "dest_tensors": {
    "out": { "size": 16 }
  },
  "constants": { "n": 4 },
  "pipelines": [
    {
      "scratch_tensors": {
        "tmp_matrix": { "size": 64 }
      },
      "kernels": [
        {
          "file": "matmul/outer_product_kernel.cu",
          "kernel": "outer_product_kernel",
          "defines": {},
          "arguments": [ "tmp_matrix", "a", "b", "n" ],
          "grid_dim": [ 1, 1, 1 ],
          "block_dim": [ 16, 1, 1 ]
        },
        {
          "file": "matmul/matvec_from_matrix_kernel.cu",
          "kernel": "matvec_from_matrix_kernel",
          "defines": {},
          "arguments": [ "out", "tmp_matrix", "x", "n" ],
          "grid_dim": [ 1, 1, 1 ],
          "block_dim": [ 4, 1, 1 ]
        }
      ]
    }
  ],
  "expected_result": "equivalent"
}
```

However, in your example the names of the tensors and constants will be different (they should match the inputs to the Python function I provide you). The json your produce will only ever have one entry in the pipelines array, but it should be an array nonetheless.


The code you are re-implementing in CUDA is the following

```python
def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    group_norm_weight,
    group_norm_bias,
    group_norm_num_groups,
    group_norm_eps,
):
    x = F.conv3d(x, conv_weight, conv_bias, stride=conv_stride, padding=conv_padding, dilation=conv_dilation, groups=conv_groups)
    x = F.group_norm(x, group_norm_num_groups, group_norm_weight, group_norm_bias, eps=group_norm_eps)
    x = x.mean(dim=[1, 2, 3, 4])
    return x
batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
num_groups = 8

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups]


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

```