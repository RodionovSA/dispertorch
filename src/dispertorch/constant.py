"""Non-dispersive (constant) models.

Two genuinely different things, kept separate rather than merged into one
ambiguous ``Constant`` (see the architecture discussion):

* :class:`ConstantIndex` --- a material with a fixed refractive index
  ``n + i k`` and no wavelength dependence. The common "just a constant" case.
* :class:`ConstantEps` --- a flat (possibly complex) permittivity. This is the
  natural additive term: the high-frequency offset ``eps_inf`` that oscillator
  models are summed onto.

Both implement :meth:`~dispertorch.base.DispersionModel.permittivity` (the
framework primitive) so they compose correctly; ``n``/``k`` derive on the base.
Coefficients are stored real and promoted internally, per the dtype convention.

``wl`` passed to any method is in **nm** (the library-wide convention); these
models ignore the wavelength value, but the argument is required for
compositional consistency with dispersive terms.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import DispersionModel
from .utils import register, register_model


@register_model("ConstantIndex")
class ConstantIndex(DispersionModel):
    """Fixed refractive index ``N = n + i k``, independent of wavelength.

    Permittivity is ``N**2``. Pass ``k`` for an absorbing constant material;
    the default ``k = 0`` is the lossless case, e.g. ``ConstantIndex(1.5)``.

    The stored constants live under ``nc``/``kc`` (also the ``state_dict``
    keys), since the base class reserves ``n``/``k`` as the derived
    ``n(wl)``/``k(wl)`` accessors. Constructor arguments stay ``n``/``k``.
    """

    def __init__(self, n, k=0.0, dtype=torch.float32):
        super().__init__()
        register(self, "nc", n, dtype)
        register(self, "kc", k, dtype)

    def permittivity(self, wl: Tensor) -> Tensor:
        N = self.nc + 1j * self.kc
        return (N * N) * torch.ones_like(wl)


@register_model("ConstantEps")
class ConstantEps(DispersionModel):
    """Flat relative permittivity ``eps = eps_re + i eps_im``.

    The default ``eps_im = 0`` gives a real constant, which is exactly the
    ``eps_inf`` background you add to oscillator terms, e.g.
    ``ConstantEps(1.0) + Lorentz(...)``.
    """

    def __init__(self, eps_re, eps_im=0.0, dtype=torch.float32):
        super().__init__()
        register(self, "eps_re", eps_re, dtype)
        register(self, "eps_im", eps_im, dtype)

    def permittivity(self, wl: Tensor) -> Tensor:
        eps = self.eps_re + 1j * self.eps_im
        return eps * torch.ones_like(wl)