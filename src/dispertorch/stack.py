"""Planar multilayer optics via recursive Fresnel (Airy) coefficients.

A compact, differentiable thin-film solver for isotropic planar stacks::

    ambient | layer 1 | layer 2 | ... | layer M | substrate [| exit medium]

The coherent part folds the stack from the substrate upwards, one interface at
a time::

    r <- (r_j + r e^{2i beta}) / (1 + r_j r e^{2i beta})
    t <- (t_j t e^{i beta})    / (1 + r_j r e^{2i beta})

An optional **incoherent substrate** (finite thickness, backside reflection,
e.g. a 0.5 mm wafer or a glass slide) combines the front coherent stack and the
back interface at the intensity level.

Conventions
-----------
* Wavelength and thicknesses in **nm**; angle of incidence ``theta`` in
  **radians**, measured in the ambient medium.
* ``exp(-i omega t)``; ``eps'' >= 0`` for absorption (library-wide).
* Fresnel coefficients are written with the admittance ``Y = k_z`` (s) or
  ``Y = k_z / eps`` (p): ``r = (Y1 - Y2)/(Y1 + Y2)``. For p this is the
  convention ``r_p = (n2 cos1 - n1 cos2)/(n2 cos1 + n1 cos2)``, so at normal
  incidence ``r_p = -r_s`` and ``rho = r_p / r_s = -1`` (``Delta = 180 deg``).
* ``ts`` and ``tp`` are ratios of transmitted to incident **electric** field
  amplitudes (``tp`` along the p unit vectors).
* Ellipsometry: ``rho = r_p / r_s = tan(psi) e^{i Delta}``. Tools that use
  ``N = n - i k`` (``exp(+i omega t)``) report ``conj(rho)``, i.e. ``-Delta``.
* The ambient medium should be transparent (real ``eps``) so that the angle
  of incidence and the incident power are well defined.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import reduce
from typing import Iterable

import torch
from torch import Tensor, nn

from .base import DispersionModel
from .constant import ConstantIndex

__all__ = ["Layer", "Stack", "StackResult"]


class Layer(nn.Module):
    """A homogeneous film: a dispersion model plus a thickness ``d`` in **nm**.

    Pass ``d`` as an ``nn.Parameter`` to fit the thickness; any other value is
    stored as a non-gradient buffer (moves with ``.to()``, saved in
    ``state_dict()``), mirroring how model coefficients are handled.
    """

    def __init__(self, model: DispersionModel, d):
        super().__init__()
        self.model = model
        _register_length(self, "d", d)


def _register_length(module: nn.Module, name: str, value) -> None:
    """Store a thickness as a Parameter (if given one) or a buffer, keeping its dtype."""
    if isinstance(value, nn.Parameter):
        setattr(module, name, value)
    else:
        dtype = value.dtype if isinstance(value, Tensor) else torch.float32
        module.register_buffer(name, torch.as_tensor(value, dtype=dtype))


@dataclass
class StackResult:
    """Output of :class:`Stack`. All tensors share the broadcast ``(wl, theta)`` shape.

    ``Rs, Rp, Ts, Tp`` are power reflectance / transmittance per polarization.
    The complex amplitudes ``rs, rp, ts, tp`` are ``None`` for an incoherent
    substrate, where no single field coefficient exists.
    """

    Rs: Tensor
    Rp: Tensor
    Ts: Tensor
    Tp: Tensor
    rs: Tensor | None = None
    rp: Tensor | None = None
    ts: Tensor | None = None
    tp: Tensor | None = None

    # -- unpolarized / derived -------------------------------------------------
    @property
    def R(self) -> Tensor:
        """Unpolarized reflectance ``(Rs + Rp) / 2``."""
        return 0.5 * (self.Rs + self.Rp)

    @property
    def T(self) -> Tensor:
        """Unpolarized transmittance ``(Ts + Tp) / 2``."""
        return 0.5 * (self.Ts + self.Tp)

    @property
    def As(self) -> Tensor:
        """s-polarized absorptance ``1 - Rs - Ts`` (whole stack incl. substrate if incoherent)."""
        return 1.0 - self.Rs - self.Ts

    @property
    def Ap(self) -> Tensor:
        """p-polarized absorptance ``1 - Rp - Tp``."""
        return 1.0 - self.Rp - self.Tp

    @property
    def A(self) -> Tensor:
        """Unpolarized absorptance ``1 - R - T``."""
        return 1.0 - self.R - self.T

    # -- ellipsometry ------------------------------------------------------------
    def _require_coherent(self, what: str) -> None:
        if self.rs is None or self.rp is None:
            raise ValueError(
                f"{what} is undefined for an incoherent substrate: the reflected "
                "beam is partially depolarized and needs a Mueller-matrix "
                "description. Use a semi-infinite substrate instead."
            )

    @property
    def rho(self) -> Tensor:
        """Complex ellipsometric ratio ``r_p / r_s = tan(psi) e^{i Delta}``."""
        self._require_coherent("rho")
        return self.rp / self.rs

    @property
    def psi(self) -> Tensor:
        """Ellipsometric angle ``psi`` in **radians**, in ``[0, pi/2]``."""
        self._require_coherent("psi")
        return torch.atan2(self.rp.abs(), self.rs.abs())

    @property
    def delta(self) -> Tensor:
        """Ellipsometric angle ``Delta = arg(r_p / r_s)`` in **radians**, in ``(-pi, pi]``."""
        return torch.angle(self.rho)


# ── numerical core ─────────────────────────────────────────────────────────────

def _kz(eps: Tensor, s2: Tensor) -> Tensor:
    """Normalized ``k_z / k_0 = sqrt(eps - eps_0 sin^2 theta)`` on the decaying branch."""
    kz = torch.sqrt(eps - s2)
    flip = (kz.imag < 0) | ((kz.imag == 0) & (kz.real < 0))
    return torch.where(flip, -kz, kz)


def _fresnel(Y1: Tensor, Y2: Tensor) -> tuple[Tensor, Tensor]:
    """Interface amplitude coefficients in terms of admittances (s: kz, p: kz/eps)."""
    den = Y1 + Y2
    return (Y1 - Y2) / den, 2.0 * Y1 / den


def _airy(Y: list[Tensor], phase: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Recursive Fresnel for media ``0..M+1``; ``phase[j] = e^{i beta}`` of medium ``j+1``.

    Returns the amplitude ``r`` and admittance-normalized ``t`` (H-field for p)
    for incidence from medium 0.
    """
    r, t = _fresnel(Y[-2], Y[-1])
    for j in range(len(Y) - 3, -1, -1):
        rj, tj = _fresnel(Y[j], Y[j + 1])
        p = phase[j]
        p2 = p * p
        den = 1.0 + rj * r * p2
        r = (rj + r * p2) / den
        t = tj * t * p / den
    return r, t


def _power_t(t: Tensor, Y_in: Tensor, Y_out: Tensor) -> Tensor:
    """Power transmittance ``Re(Y_out)/Re(Y_in) |t|^2`` (zero for evanescent output)."""
    return (Y_out.real / Y_in.real) * t.abs() ** 2


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype.is_complex:
        return dtype
    return torch.complex128 if dtype == torch.float64 else torch.complex64


# ── public module ──────────────────────────────────────────────────────────────

class Stack(nn.Module):
    """Planar isotropic multilayer: ``ambient | layers... | substrate``.

    Parameters
    ----------
    ambient:
        Incidence medium (semi-infinite, should be transparent), e.g.
        ``material("air")``.
    layers:
        Films from top (ambient side) to bottom, as :class:`Layer` objects or
        ``(model, d_nm)`` tuples. May be empty (bare substrate).
    substrate:
        Medium below the films. Semi-infinite unless ``substrate_thickness``
        is given.
    substrate_thickness:
        If set (nm), the substrate is a finite **incoherent** slab: multiple
        reflections inside it add in intensity, and light exits into
        ``exit_medium``. Use for substrates much thicker than the coherence
        length (wafers, glass slides). May be an ``nn.Parameter``.
    exit_medium:
        Medium behind an incoherent substrate (default: ``n = 1``). Only used
        with ``substrate_thickness``.

    Calling the stack with ``wl`` (nm) and optionally ``theta`` (rad) returns a
    :class:`StackResult`. ``wl`` and ``theta`` broadcast, e.g.
    ``stack(wl[:, None], theta[None, :])`` gives ``(n_wl, n_theta)`` maps.

    Notes
    -----
    Everything is differentiable w.r.t. model coefficients and thicknesses.
    Use float64 (``stack.double()`` and a float64 ``wl``) for fitting.
    Gradients w.r.t. parameters are singular exactly at grazing incidence or
    at a critical angle (``k_z = 0``), as for any Fresnel formula.

    For an incoherent substrate, power flow in an absorbing substrate uses
    ``Re(Y)|t|^2`` per pass and single-pass attenuation
    ``exp(-2 Im(k_z) d)``, the standard intensity-summation treatment.
    """

    def __init__(
        self,
        ambient: DispersionModel,
        layers: Iterable[Layer | tuple[DispersionModel, object]],
        substrate: DispersionModel,
        *,
        substrate_thickness=None,
        exit_medium: DispersionModel | None = None,
    ):
        super().__init__()
        self.ambient = ambient
        self.layers = nn.ModuleList(
            ly if isinstance(ly, Layer) else Layer(*ly) for ly in layers
        )
        self.substrate = substrate

        self.incoherent = substrate_thickness is not None
        if self.incoherent:
            _register_length(self, "substrate_thickness", substrate_thickness)
            self.exit_medium = exit_medium if exit_medium is not None else ConstantIndex(1.0)
        elif exit_medium is not None:
            raise ValueError("exit_medium requires substrate_thickness (incoherent substrate)")

    def forward(self, wl: Tensor, theta: Tensor | float = 0.0) -> StackResult:
        """Reflectance/transmittance at wavelength(s) ``wl`` [nm] and angle(s) ``theta`` [rad]."""
        wl = torch.as_tensor(wl)
        if not wl.is_floating_point():
            wl = wl.to(torch.get_default_dtype())
        theta = torch.as_tensor(theta, dtype=wl.dtype, device=wl.device)

        media = [self.ambient, *(ly.model for ly in self.layers), self.substrate]
        if self.incoherent:
            media.append(self.exit_medium)
        eps = [m.permittivity(wl) + 0j for m in media]
        cdtype = reduce(torch.promote_types, (e.dtype for e in eps), _complex_dtype(wl.dtype))
        eps = [e.to(cdtype) for e in eps]

        k0 = 2.0 * math.pi / wl
        s2 = eps[0] * torch.sin(theta) ** 2           # (n0 sin theta)^2, conserved
        kz = [_kz(e, s2) for e in eps]
        Ys = kz
        Yp = [k / e for k, e in zip(kz, eps)]

        # Medium indices: 0 ambient, 1..M films, M+1 substrate, (M+2 exit).
        M = len(self.layers)
        phase = [torch.exp(1j * k0 * kz[j + 1] * ly.d) for j, ly in enumerate(self.layers)]

        if not self.incoherent:
            rs, ts_h = _airy(Ys, phase)
            rp, tp_h = _airy(Yp, phase)
            Ts = _power_t(ts_h, Ys[0], Ys[-1])
            Tp = _power_t(tp_h, Yp[0], Yp[-1])
            # p: H-field ratio -> E-field ratio, t_E = t_H * n_in / n_out
            tp = tp_h * torch.sqrt(eps[0]) / torch.sqrt(eps[-1])
            return StackResult(
                Rs=rs.abs() ** 2, Rp=rp.abs() ** 2, Ts=Ts, Tp=Tp,
                rs=rs, rp=rp, ts=ts_h, tp=tp,
            )

        # ── incoherent substrate ────────────────────────────────────────────
        sub, ext = M + 1, M + 2
        front = slice(0, sub + 1)                      # ambient .. substrate
        # single-pass intensity attenuation through the substrate
        att = torch.exp(-2.0 * k0 * kz[sub].imag * self.substrate_thickness)
        att2 = att * att

        out = {}
        for pol, Y in (("s", Ys), ("p", Yp)):
            Yf = Y[front]
            rf, tf = _airy(Yf, phase)                  # from ambient into substrate
            rb, tb = _airy(Yf[::-1], phase[::-1])      # from substrate back to ambient
            r2, t2 = _fresnel(Y[sub], Y[ext])          # substrate -> exit

            Rf, Rb, R2 = rf.abs() ** 2, rb.abs() ** 2, r2.abs() ** 2
            Tf = _power_t(tf, Y[0], Y[sub])
            Tb = _power_t(tb, Y[sub], Y[0])
            T2 = _power_t(t2, Y[sub], Y[ext])

            den = 1.0 - Rb * R2 * att2
            out["R" + pol] = Rf + Tf * Tb * R2 * att2 / den
            out["T" + pol] = Tf * T2 * att / den

        return StackResult(**out)

    def extra_repr(self) -> str:
        kind = "incoherent substrate" if self.incoherent else "semi-infinite substrate"
        return f"n_layers={len(self.layers)}, {kind}"
