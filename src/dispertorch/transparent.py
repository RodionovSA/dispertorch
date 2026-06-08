"""Empirical, real-index dispersion models for transparent dielectrics.

Both models are **lossless** (k = 0, ε real) and use wavelength in **nm** as input,
converting internally to **µm** so that standard literature coefficients can be used
verbatim:

* ``Cauchy`` — polynomial in 1/λ²; coefficients A (dimensionless), B [µm²], C [µm⁴].
* ``Sellmeier`` — sum of resonance terms; B (dimensionless), C [µm²].
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import DispersionModel
from .utils import register, register_model

__all__ = ["Cauchy", "Sellmeier"]

# 1 nm = 1e-3 µm
_UM_PER_NM: float = 1e-3


@register_model("Cauchy")
class Cauchy(DispersionModel):
    r"""Cauchy empirical dispersion formula for transparent materials.

    .. math::

        n(\lambda) = A + \frac{B}{\lambda^2} + \frac{C}{\lambda^4}

    where λ is the wavelength in **µm**.  Permittivity is ε = n².

    Parameters
    ----------
    A:
        Zeroth-order (dimensionless) coefficient.
    B:
        Second-order coefficient [µm²].
    C:
        Fourth-order coefficient [µm⁴]; default 0 (two-term approximation).
    dtype:
        Storage dtype of the coefficients (default ``float32``).

    Notes
    -----
    ``wl`` passed to :meth:`permittivity` must be in **nm**; coefficients follow
    the standard µm convention from the literature.
    """

    def __init__(self, A, B, C=0.0, dtype=torch.float32):
        super().__init__()
        register(self, "A", A, dtype)
        register(self, "B", B, dtype)
        register(self, "C", C, dtype)

    def permittivity(self, wl: Tensor) -> Tensor:
        """Return Cauchy permittivity (real, lossless). ``wl`` in **nm**."""
        lam = wl * _UM_PER_NM         # nm → µm
        lam2 = lam ** 2
        n = self.A + self.B / lam2 + self.C / lam2 ** 2
        return n ** 2 + 0j


@register_model("Sellmeier")
class Sellmeier(DispersionModel):
    r"""Sellmeier dispersion formula for transparent optical materials.

    .. math::

        n^2(\lambda) = 1 + \sum_i \frac{B_i \lambda^2}{\lambda^2 - C_i}

    where λ is the wavelength in **µm** and :math:`C_i` are resonance wavelengths
    squared in **µm²**.

    Parameters
    ----------
    B:
        Sequence of oscillator strengths (dimensionless), length K.
    C:
        Sequence of resonance wavelengths squared [µm²], length K.
    dtype:
        Storage dtype of the coefficients (default ``float32``).

    Notes
    -----
    ``B`` and ``C`` must have the same length.

    ``wl`` passed to :meth:`permittivity` must be in **nm**; coefficients follow
    the standard µm² convention used in the literature (e.g. SCHOTT datasheet).

    The formula is only valid in the transparency window, away from any pole
    ``λ² = Cᵢ``.

    Unlike ``ConstantEps`` or ``Lorentz``, the leading ``1`` is built into
    this formula; do *not* add a ``ConstantEps(1.0)`` background separately.
    """

    def __init__(self, B, C, dtype=torch.float32):
        super().__init__()
        B_t = torch.as_tensor(B)
        C_t = torch.as_tensor(C)
        if B_t.shape != C_t.shape:
            raise ValueError(
                f"B and C must have the same shape; "
                f"got {tuple(B_t.shape)} and {tuple(C_t.shape)}"
            )
        register(self, "B", B, dtype)
        register(self, "C", C, dtype)

    def permittivity(self, wl: Tensor) -> Tensor:
        """Return Sellmeier permittivity (real, lossless). ``wl`` in **nm**."""
        lam = wl * _UM_PER_NM                        # nm → µm
        lam2 = lam ** 2
        lam2_e = lam2.unsqueeze(-1)                  # (..., 1) for broadcasting over K terms
        n2 = 1.0 + (self.B * lam2_e / (lam2_e - self.C)).sum(-1)
        return n2 + 0j
