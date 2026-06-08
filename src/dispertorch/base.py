"""Core abstractions for :mod:`dispertorch`.

Permittivity is the primitive; refractive index is derived. Susceptibilities
(hence permittivities) add, so a multi-oscillator material is a *sum* of terms
and a composite shares the single-term interface --- a lone model needs no
wrapper. Every model is an ``nn.Module`` with tensor coefficients, so models are
differentiable and ``.to(device/dtype)`` propagates through composites.

Sign convention: ``eps = eps' + i eps''`` with ``eps'' >= 0`` for absorption
(``exp(-i omega t)`` time convention), so the principal root gives ``k >= 0``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

__all__ = ["DispersionModel", "Composite"]


class DispersionModel(nn.Module):
    """Base class for dispersion models.

    Subclasses implement one primitive, :meth:`permittivity`, returning complex
    relative permittivity vs. wavelength. ``index``/``n``/``k``/``nk`` are
    derived here and should not be overridden. Models compose with ``+``.
    """

    def permittivity(self, wl: Tensor) -> Tensor:
        """Complex relative permittivity at wavelength(s) ``wl`` **[nm]** (broadcasts)."""
        raise NotImplementedError

    def forward(self, wl: Tensor) -> Tensor:
        """Alias for :meth:`permittivity` so models are callable. ``wl`` in **nm**."""
        return self.permittivity(wl)

    def index(self, wl: Tensor) -> Tensor:
        """Complex refractive index ``N = sqrt(eps)``. ``wl`` in **nm**.

        ``+ 0j`` promotes a real (lossless) permittivity to complex so the
        principal root is defined for negative real values (e.g. metals below
        the plasma frequency).
        """
        return torch.sqrt(self.permittivity(wl) + 0j)

    def n(self, wl: Tensor) -> Tensor:
        """Real part of the refractive index. ``wl`` in **nm**."""
        return self.index(wl).real

    def k(self, wl: Tensor) -> Tensor:
        """Extinction coefficient (imaginary part of the index). ``wl`` in **nm**."""
        return self.index(wl).imag

    def nk(self, wl: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(n, k)`` from a single root, avoiding recomputation. ``wl`` in **nm**."""
        N = self.index(wl)
        return N.real, N.imag

    def __add__(self, other: "DispersionModel") -> "Composite":
        """Sum two models' permittivities, flattening nested composites."""
        if not isinstance(other, DispersionModel):
            return NotImplemented
        left = list(self.terms) if isinstance(self, Composite) else [self]
        right = list(other.terms) if isinstance(other, Composite) else [other]
        return Composite(left + right)

    def __radd__(self, other: Any) -> "DispersionModel":
        """Support ``sum([...])``, whose accumulator starts at ``0``."""
        return self if other == 0 else NotImplemented


class Composite(DispersionModel):
    """A sum of dispersion terms sharing the base interface.

    Children live in an ``nn.ModuleList`` so their parameters register and
    device/dtype moves propagate; permittivity is their sum.
    """

    def __init__(self, terms: list[DispersionModel]):
        super().__init__()
        if not terms:
            raise ValueError("Composite requires at least one term")
        self.terms = nn.ModuleList(terms)

    def permittivity(self, wl: Tensor) -> Tensor:
        total = self.terms[0].permittivity(wl) + 0j
        for term in self.terms[1:]:
            total = total + term.permittivity(wl)
        return total