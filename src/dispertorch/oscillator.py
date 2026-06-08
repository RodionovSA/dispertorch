"""Oscillator dispersion models: Drude, Lorentz, Tauc-Lorentz.

All models are parameterised in photon energy (eV) but accept wavelength in nm.
The internal conversion is ``E [eV] = _HC_EV_NM / λ [nm]``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import DispersionModel
from .utils import register, register_model

__all__ = ["Drude", "Lorentz", "TaucLorentz"]

# Planck's constant times the speed of light in eV·nm
# E[eV] = _HC_EV_NM / λ[nm]
_HC_EV_NM: float = 1239.841984


def _energy(wl: Tensor) -> Tensor:
    """Convert wavelength in nm to photon energy in eV (preserves dtype/device)."""
    return _HC_EV_NM / wl


@register_model("Drude")
class Drude(DispersionModel):
    r"""Drude free-electron model.

    .. math::

        \varepsilon(E) = \frac{-\omega_p^2}{E^2 + i\gamma E}

    For positive *omega_p* and *gamma* this gives ``eps' < 0`` and
    ``eps'' ≥ 0`` (absorbing, metallic).  Add a background with
    ``ConstantEps(eps_inf) + Drude(...)``.

    Parameters
    ----------
    omega_p:
        Plasma energy [eV].
    gamma:
        Damping energy [eV].
    dtype:
        Storage dtype of the coefficients (default ``float32``).
    """

    def __init__(self, omega_p, gamma, dtype=torch.float32):
        super().__init__()
        register(self, "omega_p", omega_p, dtype)
        register(self, "gamma", gamma, dtype)

    def permittivity(self, wl: Tensor) -> Tensor:
        """Return Drude permittivity. ``wl`` in **nm**."""
        E = _energy(wl)
        return -self.omega_p ** 2 / (E ** 2 + 1j * self.gamma * E)


@register_model("Lorentz")
class Lorentz(DispersionModel):
    r"""Lorentz harmonic-oscillator model.

    .. math::

        \varepsilon(E) = \frac{A}{\omega_0^2 - E^2 - i\gamma E}

    For positive *A* and *gamma*, ``eps'' ≥ 0`` everywhere, with a peak near
    ``E = omega_0``.  Add a background with ``ConstantEps(eps_inf) + Lorentz(...)``.

    Parameters
    ----------
    A:
        Oscillator strength [eV²].
    omega_0:
        Resonance energy [eV].
    gamma:
        Damping energy [eV].
    dtype:
        Storage dtype of the coefficients (default ``float32``).
    """

    def __init__(self, A, omega_0, gamma, dtype=torch.float32):
        super().__init__()
        register(self, "A", A, dtype)
        register(self, "omega_0", omega_0, dtype)
        register(self, "gamma", gamma, dtype)

    def permittivity(self, wl: Tensor) -> Tensor:
        """Return Lorentz permittivity. ``wl`` in **nm**."""
        E = _energy(wl)
        return self.A / (self.omega_0 ** 2 - E ** 2 - 1j * self.gamma * E)


@register_model("TaucLorentz")
class TaucLorentz(DispersionModel):
    r"""Tauc-Lorentz model for amorphous semiconductors (Jellison & Modine 1996).

    The imaginary part is the product of a Tauc joint density of states and a
    Lorentz oscillator, forced to zero below the optical gap *E_g*:

    .. math::

        \varepsilon_2(E) = \begin{cases}
            \dfrac{A \omega_0 \gamma (E - E_g)^2}{[(E^2-\omega_0^2)^2 + \gamma^2 E^2]\, E}
            & E > E_g \\[6pt]
            0 & E \le E_g
        \end{cases}

    The real part is evaluated via the closed-form Kramers-Kronig transform from
    the same paper (including the 1996 erratum).

    Parameters
    ----------
    A:
        Amplitude [eV].
    omega_0:
        Lorentz-peak energy [eV]; should satisfy ``omega_0 > E_g``.
    gamma:
        Broadening [eV]; must satisfy ``gamma < 2 * omega_0`` (else ``alpha``
        is imaginary).
    E_g:
        Tauc optical gap [eV].
    dtype:
        Storage dtype of the coefficients (default ``float32``).

    Notes
    -----
    The high-frequency background ``eps_inf`` is **not** included here; compose
    it separately: ``ConstantEps(eps_inf) + TaucLorentz(...)``.

    The logarithmic term ``ln|E - E_g|`` is singular at exactly ``E = E_g``
    (zero-measure set); that point is clamped to a small positive value.
    """

    def __init__(self, A, omega_0, gamma, E_g, dtype=torch.float32):
        super().__init__()
        register(self, "A", A, dtype)
        register(self, "omega_0", omega_0, dtype)
        register(self, "gamma", gamma, dtype)
        register(self, "E_g", E_g, dtype)

    def permittivity(self, wl: Tensor) -> Tensor:
        """Return Tauc-Lorentz permittivity (eps_1 + i*eps_2). ``wl`` in **nm**."""
        E = _energy(wl)

        A, omega_0, gamma, E_g = self.A, self.omega_0, self.gamma, self.E_g

        # ── imaginary part: Tauc × Lorentz, zero below the gap ───────────────
        lorentz_den = (E ** 2 - omega_0 ** 2) ** 2 + gamma ** 2 * E ** 2
        eps_2_above = A * omega_0 * gamma * (E - E_g) ** 2 / (lorentz_den * E)
        eps_2 = torch.where(E > E_g, eps_2_above, torch.zeros_like(E))

        # ── real part: closed-form KK integral (Jellison & Modine 1996) ──────
        alpha = torch.sqrt(4 * omega_0 ** 2 - gamma ** 2)
        g2 = omega_0 ** 2 - gamma ** 2 / 2      # γ̃²; zeta4 uses γ̃² directly
        zeta4 = (E ** 2 - g2) ** 2 + alpha ** 2 * gamma ** 2 / 4

        a_ln = ((E_g ** 2 - omega_0 ** 2) * E ** 2
                + E_g ** 2 * gamma ** 2
                - omega_0 ** 2 * (omega_0 ** 2 + 3 * E_g ** 2))
        a_atan = ((E ** 2 - omega_0 ** 2) * (omega_0 ** 2 + E_g ** 2)
                  + E_g ** 2 * gamma ** 2)

        # Clamp |E − E_g| away from zero to avoid log(0) at exactly E = E_g
        abs_dE = (E - E_g).abs().clamp(min=1e-10)

        term1 = (A * gamma * a_ln / (2 * torch.pi * zeta4 * alpha * omega_0)
                 * torch.log((omega_0 ** 2 + E_g ** 2 + alpha * E_g)
                             / (omega_0 ** 2 + E_g ** 2 - alpha * E_g)))

        term2 = (A * a_atan / (torch.pi * zeta4 * omega_0)
                 * (torch.pi
                    - torch.atan((2 * E_g + alpha) / gamma)
                    + torch.atan((alpha - 2 * E_g) / gamma)))

        term3 = (4 * A * omega_0 * E_g * (E ** 2 - g2) / (torch.pi * zeta4 * alpha)
                 * (torch.atan((alpha + 2 * E_g) / gamma)
                    + torch.atan((alpha - 2 * E_g) / gamma)))

        term4 = (A * omega_0 * gamma * (E ** 2 + E_g ** 2) / (torch.pi * zeta4 * E)
                 * torch.log(abs_dE / (E + E_g)))

        term5 = (2 * A * omega_0 * gamma * E_g / (torch.pi * zeta4)
                 * torch.log(abs_dE * (E + E_g)
                             / torch.sqrt((omega_0 ** 2 - E_g ** 2) ** 2
                                          + E_g ** 2 * gamma ** 2)))

        eps_1 = term1 - term2 + term3 - term4 + term5

        return eps_1 + 1j * eps_2
