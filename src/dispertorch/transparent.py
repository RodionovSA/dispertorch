"""Empirical, real-index dispersion models for transparent dielectrics.

``Cauchy`` and ``Sellmeier`` are **lossless** (k = 0, ε real); ``CauchyUrbach``
adds a weak exponential absorption tail. All use wavelength in **nm** as input,
converting internally to **µm** so that standard literature coefficients can be used
verbatim:

* ``Cauchy`` — polynomial in 1/λ²; coefficients A (dimensionless), B [µm²], C [µm⁴].
* ``CauchyUrbach`` — Cauchy n plus k = k_amp·exp(k_exp·(E − E_band)), E in eV.
* ``Sellmeier`` — sum of resonance terms; B (dimensionless), C [µm²].
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import DispersionModel
from .utils import register, register_model

__all__ = ["Cauchy", "CauchyUrbach", "Sellmeier"]

# 1 nm = 1e-3 µm
_UM_PER_NM: float = 1e-3
# E[eV] = _HC_EV_NM / λ[nm] (same constant as oscillator.py)
_HC_EV_NM: float = 1239.841984


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


@register_model("CauchyUrbach")
class CauchyUrbach(DispersionModel):
    r"""Cauchy dispersion with an Urbach absorption tail (index-domain model).

    .. math::

        n(\lambda) &= A + \frac{B}{\lambda^2} + \frac{C}{\lambda^4} \\
        k(E) &= k_{amp} \exp\!\big(k_{exp}\,(E - E_{band})\big)

    with λ in **µm** and photon energy E in **eV** (``E = 1239.84 / λ[nm]``).
    Permittivity is ε = (n + i k)², i.e. ``n`` and ``k`` are modelled directly,
    as in Woollam CompleteEASE / Horiba, so literature parameters carry over.

    Parameters
    ----------
    A, B, C:
        Cauchy coefficients as in :class:`Cauchy` (dimensionless, µm², µm⁴).
    k_amp:
        Extinction coefficient at ``E = E_band``; default 0 (reduces to Cauchy).
    k_exp:
        Urbach exponent [1/eV].
    E_band:
        Reference (band-edge) energy [eV]; default 1239.84/400 ≈ 3.1 eV (400 nm).
    dtype:
        Storage dtype of the coefficients (default ``float32``).

    Notes
    -----
    ``k_amp`` and ``E_band`` are degenerate (only ``k_amp·exp(-k_exp·E_band)``
    matters), so keep ``E_band`` fixed (a plain value, not an ``nn.Parameter``)
    when fitting. Use bounds ``k_amp >= 0`` and ``k_exp > 0``.

    The model is not Kramers-Kronig consistent (n and k are independent), so it
    is only appropriate for weak absorption below the gap; for stronger
    absorption use :class:`~dispertorch.TaucLorentz`.

    Because it defines ``n + i k`` rather than an additive susceptibility,
    summing it with other terms via ``+`` adds its permittivity ``(n + i k)²``,
    not its ``k``.
    """

    def __init__(self, A, B, C=0.0, k_amp=0.0, k_exp=1.0,
                 E_band=_HC_EV_NM / 400.0, dtype=torch.float32):
        super().__init__()
        register(self, "A", A, dtype)
        register(self, "B", B, dtype)
        register(self, "C", C, dtype)
        register(self, "k_amp", k_amp, dtype)
        register(self, "k_exp", k_exp, dtype)
        register(self, "E_band", E_band, dtype)

    def permittivity(self, wl: Tensor) -> Tensor:
        """Return ``(n + i k)²``. ``wl`` in **nm**."""
        lam2 = (wl * _UM_PER_NM) ** 2
        n = self.A + self.B / lam2 + self.C / lam2 ** 2
        k = self.k_amp * torch.exp(self.k_exp * (_HC_EV_NM / wl - self.E_band))
        N = n + 1j * k
        return N * N


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
