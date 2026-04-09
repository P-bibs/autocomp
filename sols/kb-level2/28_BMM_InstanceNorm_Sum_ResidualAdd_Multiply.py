import torch
import torch.nn as nn
import torch.nn.functional as F

def functional_model(
    x,
    y,
    *,
    bmm_weight,
    bmm_bias,
    instance_norm_running_mean,
    instance_norm_running_var,
    instance_norm_weight,
    instance_norm_bias,
    instance_norm_use_input_stats,
    instance_norm_momentum,
    instance_norm_eps,
):
    x = F.linear(x, bmm_weight, bmm_bias)
    x = F.instance_norm(x.unsqueeze(1).unsqueeze(1), instance_norm_running_mean, instance_norm_running_var, instance_norm_weight, instance_norm_bias, use_input_stats=instance_norm_use_input_stats, momentum=instance_norm_momentum, eps=instance_norm_eps).squeeze(1).squeeze(1)
    x = x + y
    x = x * y
    return x
batch_size = 1024  # Increased batch size
in_features = 8192  # Increased input features
out_features = 8192  # Increased output features

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features), torch.rand(batch_size, out_features)]

