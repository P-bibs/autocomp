I have a collection of functional pytorch models that look like the following

```python
import torch

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
    batch_size = x.shape[0]
    device = x.device
    dtype = x.dtype
    if group_norm_bias is None:
        return torch.zeros(batch_size, device=device, dtype=dtype)
    mean_bias = group_norm_bias.mean()
    return mean_bias.expand(batch_size).to(dtype=dtype)


```

```python
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
    max_pool1_kernel_size,
    max_pool1_stride,
    max_pool1_padding,
    max_pool1_dilation,
    max_pool1_ceil_mode,
    max_pool1_return_indices,
    max_pool2_kernel_size,
    max_pool2_stride,
    max_pool2_padding,
    max_pool2_dilation,
    max_pool2_ceil_mode,
    max_pool2_return_indices,
):
    batch_size, in_channels, in_depth, in_height, in_width = x.shape
    out_channels = conv_transpose_weight.shape[0]
    
    stride = conv_transpose_stride[0] if isinstance(conv_transpose_stride, (list, tuple)) else conv_transpose_stride
    padding = conv_transpose_padding[0] if isinstance(conv_transpose_padding, (list, tuple)) else conv_transpose_padding
    output_padding = conv_transpose_output_padding[0] if isinstance(conv_transpose_output_padding, (list, tuple)) else conv_transpose_output_padding
    kernel_size = conv_transpose_weight.shape[2]
    groups = conv_transpose_groups
    dilation = conv_transpose_dilation[0] if isinstance(conv_transpose_dilation, (list, tuple)) else conv_transpose_dilation
    
    out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding
    out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding
    out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding
    
    pool1_kernel = max_pool1_kernel_size[0] if isinstance(max_pool1_kernel_size, (list, tuple)) else max_pool1_kernel_size
    pool1_stride = max_pool1_stride[0] if isinstance(max_pool1_stride, (list, tuple)) else max_pool1_stride
    pool1_pad = max_pool1_padding[0] if isinstance(max_pool1_padding, (list, tuple)) else max_pool1_padding
    pool1_dilation = max_pool1_dilation[0] if isinstance(max_pool1_dilation, (list, tuple)) else max_pool1_dilation
    
    pool2_kernel = max_pool2_kernel_size[0] if isinstance(max_pool2_kernel_size, (list, tuple)) else max_pool2_kernel_size
    pool2_stride = max_pool2_stride[0] if isinstance(max_pool2_stride, (list, tuple)) else max_pool2_stride
    pool2_pad = max_pool2_padding[0] if isinstance(max_pool2_padding, (list, tuple)) else max_pool2_padding
    pool2_dilation = max_pool2_dilation[0] if isinstance(max_pool2_dilation, (list, tuple)) else max_pool2_dilation
    
    pool1_out_depth = (out_depth + 2 * pool1_pad - pool1_dilation * (pool1_kernel - 1) - 1) // pool1_stride + 1
    pool1_out_height = (out_height + 2 * pool1_pad - pool1_dilation * (pool1_kernel - 1) - 1) // pool1_stride + 1
    pool1_out_width = (out_width + 2 * pool1_pad - pool1_dilation * (pool1_kernel - 1) - 1) // pool1_stride + 1
    
    final_depth = (pool1_out_depth + 2 * pool2_pad - pool2_dilation * (pool2_kernel - 1) - 1) // pool2_stride + 1
    final_height = (pool1_out_height + 2 * pool2_pad - pool2_dilation * (pool2_kernel - 1) - 1) // pool2_stride + 1
    final_width = (pool1_out_width + 2 * pool2_pad - pool2_dilation * (pool2_kernel - 1) - 1) // pool2_stride + 1
    
    output = torch.zeros(batch_size, 1, final_depth, final_height, final_width, device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op_forward(
        x, conv_transpose_weight, conv_transpose_bias, output,
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        kernel_size, stride, padding, output_padding, groups, dilation,
        pool1_kernel, pool1_stride, pool1_pad, pool1_dilation,
        pool2_kernel, pool2_stride, pool2_pad, pool2_dilation
    )
    
    return output
```

See full set in output_collated/level2, but there are a lot.7

Notice that these functions might contain arbitrary python code, but I only care about very specific function calls:
* pytorch operations (for example, torch.zeros which creates a pytorch tensor, or functions like matmul and convolution that transform pytorch tensors)
* calls to custom CUDA kernels (like fused_ext.fused_op_forward)

I want to write a tracer that can run a function like this and produce an ordered list of each torch operation or custom CUDA kernel that is called, along with the arguments that are passed to them. In the case of CUDA kernels, I also want the PTX code and launch grid configuration that is being run.

I propose a method for doing each of these, but you can adjust if you think there is a better way.

# PyTorch Operations:
I recommend you take advantage of `__torch_function__` to pass in objects that act like torch tensors but also log whenever a function is called on them. If you don't think this is the write approach, you can try something else.
For function like `torch.zeros`, one solution is to list all of them and then monkeypatch them in the torch namespace to log when they're called.
You can treat reshaping and view operations and similar as no-ops, so they're allowable.

# Cuda Operations
To trace CUDA calls, you can use LD_PRELOAD to hook __cudaRegisterFunction, cudaLaunchKernel, etc. For inspiration, look at cuda_trace_module_preload.cpp, which should contain everything you need to view kernel launches and inspect their code, argument values, argument types, and grid configuration.

To identify the arguments, you can look at the pointers that are passed into the kernel and see if it matches any tensor you saw created by PyTorch or as input to the function. If it isn't, exit and report an error. This also means you'll need to keep track of a map from pointer address to tensor name (for input arguments, you have a name for the tensor. For temporary tensors created by torch.zeros or the like, don't try and analyze the AST to get the tensor name, just use a generated name).
For arguments that are not pointers like floats and ints, you can record their value directly and save it. However, make sure its saved as the right format (floats should be floats and ints should be ints).
If we  see a pointer argument that we don't recognize (not an input tensor or created by a torch call we've seen) then that is an error.

# Postprocessing

Once this process is completed and the list of operations is produced, I want to do some postprocessing.
Programs I consider well-formed only use pytorch operations to make new tensors, such as `torch.zeros`, `torch.ones`, or `torch.random`. If there are any torch operators used that are not like this (such as matmul or convolution), the program should be rejected and the run should abort.
If this is not the case, we continue to generate a JSON output. As input to the script, you will get a json file that looks like this:
```
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
Our goal is to take the behavior of the python function we just analyzed and insert it as an additional pipeline to the list of pipelines already in this json.
In order to do this, we need to make sure the set of source tensors + dest tensors in the json file is exactly the same as the set of arguments we have to our function. If not, we should report an error. There is always exaclty one output tensor, the tensor that is returned.
Beyond this, we need to make a new pipeline entry and append it to the existing list. Our pipelines should have a scratch tensor for any call to an intermediate pytorch function that creates a tensor (like torch.zeros) (these torch operations should not be represented otherwise, such as a pipeline entry).
Instead of having a filepath where the kernel is, our pipeline should express its kernels by having a field `"source": "ptx"` and a field `"ptx"` that contains the full PTX code as a string.

Once the script has produced the new json, it should be written to an output file. It should not touch the input json in anyway.

Finally, I want to make sure the underlying Python code is mostly deterministic, so tensor sizes and scalar arguments to CUDA kernels shouldn't change (the exact addresses of pointers can change). To make sure of this, run the whole pipeline twice and make sure you get the same json result. Make this option configurable via a cli flag.

Put your output in a directory called `trace_functional_to_cuda`