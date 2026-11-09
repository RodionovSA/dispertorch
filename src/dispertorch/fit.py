"""Least-squares fitting of dispersion models and multilayer stacks to data.

Describe each dataset as a :class:`Measurement` (what was measured, at which
wavelengths and angles, with what uncertainty), then call :func:`fit` on the
system --- a :class:`~dispertorch.Stack` or a bare
:class:`~dispertorch.DispersionModel`. Datasets can mix quantities and angles,
e.g. ellipsometry at 65/70/75 deg together with normal-incidence R and T.

Free parameters are the system's ``nn.Parameter``s with ``requires_grad=True``
(or an explicit ``params=`` subset). They are updated **in place**, like a
torch optimizer, and a :class:`FitResult` with statistics is returned.

The objective is ``chi^2 = sum_i ((model_i - data_i) / sigma_i)^2`` over all
measurements. Give realistic ``sigma`` per dataset: it sets the relative weight
of, say, ``R`` (0..1) against ``psi`` in degrees, and makes ``stderr``
meaningful.

Methods
-------
``"lm"`` (default)
    Levenberg--Marquardt with Marquardt diagonal scaling and the Nielsen damping
    update. The Jacobian comes from :func:`torch.func.jacfwd` (falls back to
    ``jacrev``). The right tool for this problem class.
``"lbfgs"``
    :class:`torch.optim.LBFGS` with strong-Wolfe line search on ``chi^2 / 2``.
``"adam"``
    :class:`torch.optim.Adam`; only sensible for large or noisy problems.

All methods run internally in ``dtype`` (float64 by default) regardless of the
system's own dtype, and write the result back in the parameters' dtypes.
Bounds are enforced by projection (clamping).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.func import functional_call, jacfwd, jacrev

from .base import DispersionModel

__all__ = ["Measurement", "FitResult", "fit"]

STACK_KINDS = ("R", "Rs", "Rp", "T", "Ts", "Tp", "A", "As", "Ap", "psi", "delta")
MODEL_KINDS = ("n", "k", "eps1", "eps2")
_ANGULAR = ("psi", "delta")


@dataclass
class Measurement:
    """One dataset: ``data`` of quantity ``kind`` at wavelengths ``wl`` and angles ``theta``.

    Parameters
    ----------
    kind:
        For a :class:`~dispertorch.Stack`: ``"R" "Rs" "Rp" "T" "Ts" "Tp" "A"
        "As" "Ap" "psi" "delta"`` (unpolarized ``R``/``T``/``A`` are s/p
        averages). For a :class:`~dispertorch.DispersionModel`: ``"n" "k"
        "eps1" "eps2"``. Or a callable ``f(system, wl, theta) -> Tensor`` for
        anything else.
    wl:
        Wavelengths in **nm**.
    data:
        Measured values; must have the broadcast shape of ``wl`` and ``theta``.
        For variable-angle data use e.g. ``wl[:, None]``, ``theta[None, :]`` and
        ``data`` of shape ``(n_wl, n_theta)``.
    theta:
        Angle(s) of incidence (ignored for model kinds). Radians unless
        ``degrees=True``.
    sigma:
        Uncertainty per point (scalar or broadcastable), in the units of ``data``.
    degrees:
        If ``True``, ``theta`` and angular data (``psi``, ``delta``) are in
        degrees.
    name:
        Optional label used in :class:`FitResult`.

    ``delta`` residuals are wrapped into ``(-180, 180]`` deg, so data and model
    may differ by a full turn.
    """

    kind: str | Callable[[nn.Module, Tensor, Tensor], Tensor]
    wl: Tensor
    data: Tensor
    theta: Tensor | float = 0.0
    sigma: Tensor | float = 1.0
    degrees: bool = False
    name: str | None = None

    def __post_init__(self):
        if isinstance(self.kind, str) and self.kind not in STACK_KINDS + MODEL_KINDS:
            raise ValueError(
                f"unknown kind {self.kind!r}; expected one of {STACK_KINDS + MODEL_KINDS} "
                "or a callable f(system, wl, theta)"
            )
        if self.name is None:
            self.name = self.kind if isinstance(self.kind, str) else getattr(
                self.kind, "__name__", "custom")


@dataclass
class FitResult:
    """Outcome of :func:`fit`. Parameter values are also written into the system.

    ``cost`` is ``chi^2``; ``reduced_chi2 = chi^2 / (N - P)`` (~1 for a good fit
    with correct ``sigma``). ``stderr`` and ``correlation`` come from the
    covariance ``reduced_chi2 * (J^T J)^{-1}`` at the solution; with
    unknown/unit ``sigma`` this is the usual rescaled estimate.
    """

    success: bool
    message: str
    method: str
    n_iter: int
    cost: float
    reduced_chi2: float
    n_data: int
    params: dict[str, Tensor]
    stderr: dict[str, Tensor]
    labels: list[str]
    correlation: Tensor
    residuals: dict[str, Tensor] = field(repr=False)
    history: list[float] = field(repr=False)

    def summary(self) -> str:
        """Human-readable table of the fitted parameters and fit statistics."""
        lines = [
            f"method={self.method}  success={self.success}  iterations={self.n_iter}",
            f"{self.message}",
            f"chi2={self.cost:.6g}  reduced chi2={self.reduced_chi2:.6g}  "
            f"N={self.n_data}  P={len(self.labels)}",
            "",
            f"{'parameter':<40} {'value':>14} {'stderr':>12}",
        ]
        for name, val in self.params.items():
            v, e = val.reshape(-1), self.stderr[name].reshape(-1)
            for i in range(v.numel()):
                label = name if v.numel() == 1 else f"{name}[{i}]"
                lines.append(f"{label:<40} {v[i].item():>14.6g} {e[i].item():>12.3g}")
        return "\n".join(lines)


# ── evaluation ─────────────────────────────────────────────────────────────────

class _Evaluator(nn.Module):
    """Wraps the system so ``functional_call`` can swap in parameter values."""

    def __init__(self, system: nn.Module, measurements: Sequence[Measurement], dtype, device):
        super().__init__()
        self.system = system
        self.meas = []

        def prep(v):
            # convert on the source device first: MPS cannot hold float64 at all
            return torch.as_tensor(v).to(dtype).to(device)

        for m in measurements:
            wl = prep(m.wl)
            theta = prep(m.theta)
            if m.degrees:
                theta = torch.deg2rad(theta)
            data = prep(m.data)
            sigma = prep(m.sigma)
            scale = 180.0 / math.pi if (m.degrees and m.kind in _ANGULAR) else 1.0
            self.meas.append((m, wl, theta, data, sigma, scale))

    def _model(self, m: Measurement, wl: Tensor, theta: Tensor) -> Tensor:
        sys = self.system
        if callable(m.kind):
            return m.kind(sys, wl, theta)
        if m.kind in MODEL_KINDS:
            if not isinstance(sys, DispersionModel):
                raise ValueError(f"kind {m.kind!r} needs a DispersionModel system, "
                                 f"got {type(sys).__name__}")
            if m.kind in ("n", "k"):
                return getattr(sys, m.kind)(wl)
            eps = sys.permittivity(wl) + 0j
            return eps.real if m.kind == "eps1" else eps.imag
        if isinstance(sys, DispersionModel):
            raise ValueError(f"kind {m.kind!r} needs a Stack system, got a DispersionModel")
        return getattr(sys(wl, theta), m.kind)

    def forward(self) -> list[Tensor]:
        out = []
        for m, wl, theta, data, sigma, scale in self.meas:
            model = self._model(m, wl, theta)
            if model.shape != data.shape:
                raise ValueError(
                    f"measurement {m.name!r}: data shape {tuple(data.shape)} does not match "
                    f"model output shape {tuple(model.shape)} (broadcast of wl and theta)"
                )
            if m.kind == "delta":
                d = model - data / scale                              # radians
                diff = torch.atan2(torch.sin(d), torch.cos(d)) * scale  # wrapped
            else:
                diff = model * scale - data
            out.append((diff / sigma).reshape(-1))
        return out


# ── optimizers ─────────────────────────────────────────────────────────────────

def _levenberg_marquardt(f, x, lo, hi, max_iter, tol, verbose):
    def project(z):
        return torch.clamp(z, lo, hi)

    jac = _jacobian_fn(f, x)

    r = f(x)
    cost = float(r @ r)
    history = [cost]
    lam, nu = 1e-3, 2.0
    message, success, it = "maximum number of iterations reached", False, 0
    eye_eps = torch.finfo(x.dtype).eps

    for it in range(1, max_iter + 1):
        J = jac(x)
        A = J.T @ J
        g = J.T @ r
        if g.abs().max() <= tol * max(cost, eye_eps):
            message, success = "gradient below tolerance", True
            break
        D = torch.clamp(torch.diagonal(A), min=eye_eps * torch.diagonal(A).max().clamp(min=1.0))

        accepted = False
        for _ in range(50):
            try:
                step = torch.linalg.solve(A + lam * torch.diag(D), -g)
            except RuntimeError:
                step = torch.linalg.lstsq(A + lam * torch.diag(D), -g[:, None]).solution[:, 0]
            x_new = project(x + step)
            step = x_new - x
            r_new = f(x_new)
            cost_new = float(r_new @ r_new)
            pred = -float(2 * step @ g + step @ (A @ step))     # predicted chi2 decrease
            if math.isfinite(cost_new) and cost_new < cost:
                rho = (cost - cost_new) / pred if pred > 0 else 1.0
                lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                nu = 2.0
                accepted = True
                break
            lam *= nu
            nu *= 2.0
            if lam > 1e16:
                break

        if not accepted:
            message, success = "no step reduces chi2 (converged to tolerance of the model)", True
            break

        dcost = cost - cost_new
        small_step = float(step.norm()) <= tol * (float(x.norm()) + tol)
        x, r, cost = x_new, r_new, cost_new
        history.append(cost)
        if verbose:
            print(f"[lm] iter {it:4d}  chi2={cost:.8g}  lambda={lam:.2e}")
        if dcost <= tol * history[-2]:
            message, success = "relative chi2 change below tolerance", True
            break
        if small_step:
            message, success = "relative step size below tolerance", True
            break

    return x, history, it, success, message


def _jacobian_fn(f, x):
    """``jacfwd(f)`` (cheap for many residuals, few parameters), else ``jacrev(f)``."""
    jac = jacfwd(f)
    try:
        jac(x)
    except Exception:            # op without forward-mode AD -> reverse mode
        jac = jacrev(f)
    return jac


def _variable_scale(f, x, method):
    """Per-parameter scale ``s`` for the substitution ``x = x0 + s * u``.

    Parameters differ by orders of magnitude (d ~ 1e2 nm, Cauchy B ~ 1e-3 um^2),
    which stalls first-order/quasi-Newton methods. L-BFGS gets Jacobi
    preconditioning (unit curvature per variable, like LM's Marquardt scaling);
    Adam gets relative scaling, so ``lr`` is a fractional step.
    """
    if method == "lbfgs":
        J = _jacobian_fn(f, x)(x)
        s = 1.0 / torch.sqrt((J * J).sum(0))
    else:
        s = x.abs()
    ok = torch.isfinite(s) & (s > 0)
    return torch.where(ok, s, torch.ones_like(s))


def _torch_optimizer(f, x, lo, hi, method, max_iter, tol, lr, verbose):
    scale = _variable_scale(f, x, method).detach()
    x0 = x.detach()
    z = torch.zeros_like(x0, requires_grad=True)      # scaled offset from x0

    def to_x(u):
        return torch.clamp(x0 + scale * u, lo, hi)

    def loss_fn():
        r = f(to_x(z))
        return 0.5 * (r @ r)

    history = [2 * float(loss_fn().detach())]
    if method == "lbfgs":
        opt = torch.optim.LBFGS([z], lr=1.0, max_iter=max_iter, max_eval=2 * max_iter,
                                tolerance_grad=tol, tolerance_change=tol,
                                history_size=50, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            history.append(2 * float(loss.detach()))
            return loss

        opt.step(closure)
        n_iter = opt.state[z].get("n_iter", 0)
        success = n_iter < max_iter
        message = "converged (gradient or change below tolerance)" if success \
            else "maximum number of iterations reached"
    else:
        opt = torch.optim.Adam([z], lr=lr)
        success, message, n_iter = False, "maximum number of iterations reached", max_iter
        window = 10                       # Adam steps are noisy: compare over a window
        losses = []
        best_loss, best_z = math.inf, z.detach().clone()
        for it in range(1, max_iter + 1):
            opt.zero_grad()
            loss = loss_fn()              # chi2/2 at the point *before* this step
            loss.backward()
            if float(loss.detach()) < best_loss:   # Adam oscillates near the minimum:
                best_loss = float(loss.detach())   # keep the best point visited
                best_z = z.detach().clone()
            opt.step()
            losses.append(2 * float(loss.detach()))
            history.append(losses[-1])
            if verbose and it % 100 == 0:
                print(f"[adam] iter {it:5d}  chi2={losses[-1]:.8g}")
            if len(losses) > window and \
                    abs(losses[-window - 1] - losses[-1]) <= tol * losses[-window - 1]:
                success, message, n_iter = True, "relative chi2 change below tolerance", it
                break
        history.pop(1)                    # duplicate of the initial chi2
        with torch.no_grad():
            r_last = f(to_x(z))
            if 0.5 * float(r_last @ r_last) > best_loss:
                z.copy_(best_z)

    x_final = to_x(z.detach())
    r = f(x_final)
    history.append(float(r @ r))
    return x_final, history, n_iter, success, message


# ── public API ─────────────────────────────────────────────────────────────────

def _resolve(system: nn.Module, q: nn.Parameter | str) -> str:
    """Name of a parameter given by object or by name."""
    names = dict(system.named_parameters())
    if isinstance(q, str):
        if q not in names:
            raise ValueError(f"system has no parameter named {q!r}; "
                             f"available: {list(names)}")
        return q
    for n, p in names.items():
        if p is q:
            return n
    raise ValueError(
        "parameter object is not part of the system (note: models replace a Parameter "
        "whose dtype differs from their own; pass the parameter name instead)"
    )


def fit(
    system: nn.Module,
    measurements: Measurement | Iterable[Measurement],
    *,
    method: str = "lm",
    max_iter: int = 200,
    tol: float = 1e-10,
    params: Iterable[nn.Parameter | str] | None = None,
    bounds: dict[nn.Parameter | str, tuple] | None = None,
    lr: float = 1e-2,
    dtype: torch.dtype | None = None,
    verbose: bool = False,
) -> FitResult:
    """Fit ``system`` parameters to one or more measurements (weighted least squares).

    Parameters
    ----------
    system:
        A :class:`~dispertorch.Stack` or :class:`~dispertorch.DispersionModel`
        (any ``nn.Module`` works with callable measurement kinds).
    measurements:
        One :class:`Measurement` or a list of them; residuals are concatenated.
    method:
        ``"lm"`` (Levenberg--Marquardt, default), ``"lbfgs"`` or ``"adam"``.
    max_iter:
        Maximum iterations (LM: Jacobian evaluations; L-BFGS/Adam: optimizer
        iterations).
    tol:
        Convergence tolerance: relative change of ``chi^2`` and relative step
        size (LM), gradient/change tolerance (L-BFGS), relative ``chi^2``
        change (Adam).
    params:
        Parameters to fit, as ``nn.Parameter`` objects or names from
        ``system.named_parameters()`` (e.g. ``"layers.0.d"``). Default: all
        parameters with ``requires_grad=True``. Others stay fixed.
    bounds:
        ``{parameter_or_name: (low, high)}``; ``low``/``high`` scalars or tensors
        shaped like the parameter, ``None`` for unbounded. Enforced by projection.
    lr:
        Learning rate for ``"adam"``, as a **relative** step: parameters are
        scaled by their initial magnitude, so ``1e-2`` means ~1 % per step.
        (L-BFGS is preconditioned by ``diag(J^T J)`` at the start point.)
    dtype:
        Working precision. Default: float64, or float32 on devices without
        float64 support (Apple MPS).
    verbose:
        Print progress.

    Device handling: the fit runs on the device of the system's parameters;
    measurement arrays may live anywhere and are moved there. Parameters are
    written back on their own device and dtype. ``FitResult`` tensors are
    small summaries and are returned on CPU; the covariance is always computed
    on CPU in float64.
    """
    method = method.lower()
    if method not in ("lm", "lbfgs", "adam"):
        raise ValueError(f"unknown method {method!r}; use 'lm', 'lbfgs' or 'adam'")
    if isinstance(measurements, Measurement):
        measurements = [measurements]
    measurements = list(measurements)
    if not measurements:
        raise ValueError("at least one measurement is required")

    named = [(n, p) for n, p in system.named_parameters() if p.requires_grad]
    if params is not None:
        wanted = {_resolve(system, q) for q in params}
        named = [(n, p) for n, p in named if n in wanted]
        if len(named) != len(wanted):
            raise ValueError("params contains entries that are not trainable parameters "
                             "of the system")
    if not named:
        raise ValueError("no free parameters: wrap coefficients/thicknesses in nn.Parameter")

    devices = {p.device for _, p in named}
    if len(devices) > 1:
        raise ValueError(f"fitted parameters live on several devices {devices}; "
                         "move the system with .to(device) first")
    device = devices.pop()
    if dtype is None:
        dtype = torch.float32 if device.type == "mps" else torch.float64
    elif dtype == torch.float64 and device.type == "mps":
        raise ValueError("MPS does not support float64; pass dtype=torch.float32 "
                         "or fit on CPU")

    evaluator = _Evaluator(system, measurements, dtype, device)
    names = [n for n, _ in named]
    shapes = [p.shape for _, p in named]
    sizes = [p.numel() for _, p in named]
    x0 = torch.cat([p.detach().reshape(-1).to(dtype) for _, p in named])

    def as_bound(v, shape):
        return torch.as_tensor(v).to(dtype).to(device).expand(shape).reshape(-1)

    lo = torch.full_like(x0, -math.inf)
    hi = torch.full_like(x0, math.inf)
    if bounds:
        idx = {n: i for i, (n, _) in enumerate(named)}
        offsets = [0, *torch.cumsum(torch.tensor(sizes), 0).tolist()]
        for q, (b_lo, b_hi) in bounds.items():
            n = _resolve(system, q)
            if n not in idx:
                raise ValueError(f"bounds given for {n!r}, which is not being fitted")
            i = idx[n]
            sl = slice(offsets[i], offsets[i + 1])
            if b_lo is not None:
                lo[sl] = as_bound(b_lo, shapes[i])
            if b_hi is not None:
                hi[sl] = as_bound(b_hi, shapes[i])
    x0 = torch.clamp(x0, lo, hi)

    # Everything else (buffers) is also evaluated at working precision.
    fixed = {f"system.{n}": b.to(dtype) if b.is_floating_point() else b
             for n, b in system.named_buffers()}
    fixed.update({f"system.{n}": p.detach().to(dtype)
                  for n, p in system.named_parameters() if n not in names})

    def unflatten(x):
        return {f"system.{n}": t.reshape(s)
                for n, t, s in zip(names, torch.split(x, sizes), shapes)}

    def residual_parts(x):
        return functional_call(evaluator, {**fixed, **unflatten(x)}, ())

    def residual(x):
        return torch.cat(residual_parts(x))

    r0 = residual(x0)
    if not torch.isfinite(r0).all():
        raise ValueError("non-finite residuals at the initial parameters")

    if method == "lm":
        x, history, n_iter, success, message = _levenberg_marquardt(
            residual, x0, lo, hi, max_iter, tol, verbose)
    else:
        x, history, n_iter, success, message = _torch_optimizer(
            residual, x0, lo, hi, method, max_iter, tol, lr, verbose)

    # write back
    with torch.no_grad():
        for (_, p), t in zip(named, torch.split(x, sizes)):
            p.copy_(t.reshape(p.shape).to(p.dtype))

    # statistics at the solution
    parts = residual_parts(x)
    r = torch.cat(parts).detach()
    cost = float(r @ r)
    n_data, n_par = r.numel(), x.numel()
    dof = n_data - n_par
    red = cost / dof if dof > 0 else math.inf
    J = _jacobian_fn(residual, x)(x)
    # P x P problem: do it on CPU in float64 (pinv/SVD is unsupported on MPS).
    JTJ = (J.T @ J).detach().cpu().to(torch.float64)
    cov = torch.linalg.pinv(JTJ) * (red if math.isfinite(red) else math.nan)
    err = torch.sqrt(torch.clamp(torch.diagonal(cov), min=0.0))
    denom = torch.outer(err, err)
    corr = torch.where(denom > 0, cov / denom, torch.zeros_like(cov))
    x = x.detach().cpu()

    labels = []
    for n, s in zip(names, sizes):
        labels += [n] if s == 1 else [f"{n}[{i}]" for i in range(s)]
    params_out = {n: t.reshape(s).clone() for n, t, s in zip(names, torch.split(x, sizes), shapes)}
    stderr_out = {n: t.reshape(s).clone() for n, t, s in zip(names, torch.split(err, sizes), shapes)}
    resid_out = {}
    for m, part in zip(measurements, parts):
        key, i = m.name, 2
        while key in resid_out:
            key, i = f"{m.name}_{i}", i + 1
        resid_out[key] = part.detach().cpu()

    return FitResult(
        success=success, message=message, method=method, n_iter=n_iter,
        cost=cost, reduced_chi2=red, n_data=n_data,
        params=params_out, stderr=stderr_out, labels=labels,
        correlation=corr.detach(), residuals=resid_out, history=history,
    )
