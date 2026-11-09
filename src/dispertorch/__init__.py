"""dispertorch — differentiable refractive-index dispersion models."""

from .base import Composite, DispersionModel
from .constant import ConstantEps, ConstantIndex
from .data import list_materials, material
from .oscillator import Drude, Lorentz, TaucLorentz
from .stack import Layer, Stack, StackResult
from .transparent import Cauchy, Sellmeier
from .utils import MODEL_REGISTRY, register, register_model

__all__ = [
    # base
    "DispersionModel",
    "Composite",
    # non-dispersive
    "ConstantIndex",
    "ConstantEps",
    # oscillators
    "Drude",
    "Lorentz",
    "TaucLorentz",
    # transparent / empirical
    "Cauchy",
    "Sellmeier",
    # planar multilayers
    "Layer",
    "Stack",
    "StackResult",
    # material presets
    "list_materials",
    "material",
    # helpers
    "register",
    "register_model",
    "MODEL_REGISTRY",
]
