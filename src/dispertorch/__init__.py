"""dispertorch — differentiable refractive-index dispersion models."""

from .base import Composite, DispersionModel
from .constant import ConstantEps, ConstantIndex
from .data import list_materials, material
from .fit import FitResult, Measurement, fit
from .oscillator import Drude, Lorentz, TaucLorentz
from .stack import Layer, Stack, StackResult
from .transparent import Cauchy, CauchyUrbach, Sellmeier
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
    "CauchyUrbach",
    "Sellmeier",
    # planar multilayers
    "Layer",
    "Stack",
    "StackResult",
    # fitting
    "Measurement",
    "FitResult",
    "fit",
    # material presets
    "list_materials",
    "material",
    # helpers
    "register",
    "register_model",
    "MODEL_REGISTRY",
]
