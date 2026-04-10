There is currently a file functional_to_lambda.py that takes a file from the KernelBenchFunctional directory that looks like this:
```python
# END EVAL UTILS

def functional_model(
    x,
    *,
    gemm_weight,
    bias,
):
    x = F.linear(x, gemm_weight, None)
    x = x + bias
    x = torch.relu(x)
    return x
batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

def get_init_inputs():
    return [in_features, out_features, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_features)]

```

And converts it to a variety of formats based on a `--backend` flag. I would like you to add to the `--backend cpp` mode to also output a file like this:
```json
// 76_Gemm_Add_ReLU.json
{
  "source_tensors": {
    "x": { "size": 8388608 },
    "gemm_weight": { "size": 67108864 },
    "bias": { "size": 8192 }
  },
  "dest_tensors": {
    "out": { "size": 8388608 }
  },
  "constants": {  },
  "pipelines": [
    {
      "scratch_tensors": {},
      "kernels": [
        {
          "source": "spec",
          "file": "76_Gemm_Add_ReLU.cpp",
        }
      ]
    }
  ],
  "expected_result": "equivalent"
}
```

It should be in exactly this format, but with the correct tensor sizes based on analysis of the program and the proper `cpp` filename based on the provided output path. Note that there is always exactly one output tensor and the name does not matter.

Interview me.