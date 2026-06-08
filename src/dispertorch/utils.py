"""Shared helpers for :mod:`dispertorch`: the model registry and coefficient
registration."""

import torch
from torch import nn

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import DispersionModel

MODEL_REGISTRY: dict[str, type["DispersionModel"]] = {}


def register_model(name: str):
    """Register a :class:`DispersionModel` subclass under ``name``."""
    def deco(cls: type["DispersionModel"]) -> type["DispersionModel"]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered")
        MODEL_REGISTRY[name] = cls
        return cls
    return deco


def register(module: nn.Module, name: str, value, dtype=torch.float32) -> None:
    """Register `value` on `module` under `name`.

    If `value` is an nn.Parameter it becomes an optimizable parameter;
    otherwise it is stored as a (non-gradient) buffer that still moves
    with .to() and is saved in state_dict().
    """
    if isinstance(value, nn.Parameter):
        if value.dtype != dtype:
            value = nn.Parameter(value.detach().to(dtype),
                                  requires_grad=value.requires_grad)
        setattr(module, name, value)
    else:
        module.register_buffer(name, torch.as_tensor(value, dtype=dtype))