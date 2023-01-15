"""Implementation of different utility functions for adapter layers."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


class GEGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, -1)
        return self.gelu(x1) * x2


class LoraLinear(nn.Module):
    def __init__(self, weight, bias=None, adapter_dim=0, parallel=False,
                 batchensemble=False, adapter_bias=False, init_svd=False):
        assert isinstance(bias, nn.Parameter) or bias is None
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.bias = bias
        self.parallel = parallel
        self.init_svd = init_svd

        if adapter_dim > 0:
            self.adapter_down = nn.Linear(
                self.in_features, adapter_dim, bias=False)
            self.adapter_up = nn.Linear(
                adapter_dim, self.out_features, bias=False)

            self.adapter_dim = adapter_dim
            if adapter_bias:
                self.adapter_bias = nn.Parameter(
                    torch.zeros(self.out_features))
            else:
                self.adapter_bias = None
            if self.init_svd:
                U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
                self.adapter_up.weight.data = U[:,
                                                :self.adapter_dim]  @ torch.diag(S[:self.adapter_dim])
                self.adapter_down.weight.data = Vh[:self.adapter_dim, :]
                weight.data = weight.data - \
                    (U[:, :self.adapter_dim]  @ torch.diag(S[:self.adapter_dim])
                     @ Vh[:self.adapter_dim, :])
            else:
                nn.init.zeros_(self.adapter_up.weight)
        else:
            self.adapter_down = None

        if batchensemble:
            self.adapter_before = nn.Parameter(torch.zeros(self.in_features))
            self.adapter_after = nn.Parameter(torch.zeros(self.out_features))
            if adapter_bias:
                self.adapter_bias2 = nn.Parameter(
                    torch.zeros(self.out_features))
            else:
                self.adapter_bias2 = None
        else:
            self.adapter_before = None
            self.adapter_after = None
            self.adapter_bias2 = None

    def forward(self, input):
        if self.adapter_before is not None:
            input = input * (1. + 128.0**0.5 * self.adapter_before)
        out = F.linear(input, self.weight, self.bias)
        if self.adapter_after is not None:
            out = out * (1. + 128.0**0.5 * self.adapter_after)
        if self.adapter_bias2 is not None:
            out = out + self.adapter_bias2

        if self.parallel and self.adapter_down is not None:
            input = self.adapter_down(input)
            out = out + (128.0 / self.adapter_dim) * self.adapter_up(input)
            if self.adapter_bias is not None:
                out = out + self.adapter_bias
        if not self.parallel and self.adapter is not None:
            out = out + self.adapter_bias + \
                (128.0 / self.adapter_dim) * self.adapter(out)
            if self.adapter_bias is not None:
                out = out + self.adapter_bias

        return out

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "LoraLinear":
        return cls(linear.weight, linear.bias, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"


def lorafy_(model, config):
    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and name in config.layer_list.split(","):
                for layer in config.layers:
                    if config.rewrite_module_tmp.format(layer) in name:
                        print(name, child)
                        setattr(module, name, LoraLinear.from_linear(
                            child, adapter_dim=config.adapter_size))
                        # parallel=config.parallel, batchensemble=config.batchensemble,
                        # adapter_bias=config.lora_bias, down_scale=config.down_scale,
                        # up_scale=config.up_scale, init_svd=False))
