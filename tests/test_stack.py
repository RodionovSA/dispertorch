"""Tests for the planar multilayer solver (recursive Fresnel + incoherent substrate)."""

import math

import pytest
import torch
import torch.nn as nn

from dispertorch import ConstantIndex, Layer, Stack, material

F64 = torch.float64
WL0 = 600.0


def ci(n, k=0.0):
    return ConstantIndex(n, k, dtype=F64)


def wl(*vals):
    return torch.tensor(vals or (WL0,), dtype=F64)


def close(a, b, tol=1e-10):
    """Elementwise |a - b| <= tol, broadcasting scalars; works for real and complex."""
    a = torch.as_tensor(a)
    b = b if isinstance(b, torch.Tensor) else torch.tensor(b, dtype=torch.complex128)
    diff = (a.to(torch.complex128) - b.to(torch.complex128)).abs()
    assert diff.max().item() <= tol, f"max |a-b| = {diff.max().item():.3e} > {tol:g}\na={a}\nb={b}"


# ── single interface vs closed-form Fresnel ────────────────────────────────

class TestInterface:
    def test_normal_incidence(self):
        res = Stack(ci(1.0), [], ci(1.5))(wl())
        R = ((1.5 - 1.0) / 2.5) ** 2
        close(res.Rs, R)
        close(res.Rp, R)
        close(res.Ts, 1 - R)
        close(res.rho, -1.0 + 0j)   # r_p = -r_s at normal incidence (documented)

    def test_oblique_matches_fresnel(self):
        n1, n2, th = 1.0, 1.7, math.radians(50)
        c1 = math.cos(th)
        c2 = math.sqrt(1 - (n1 * math.sin(th) / n2) ** 2)
        rs = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
        rp = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
        ts = 2 * n1 * c1 / (n1 * c1 + n2 * c2)
        tp = 2 * n1 * c1 / (n2 * c1 + n1 * c2)
        res = Stack(ci(n1), [], ci(n2))(wl(), th)
        close(res.rs.real, rs)
        close(res.rp.real, rp)
        close(res.ts.real, ts)
        close(res.tp.real, tp)
        close(res.Ts, n2 * c2 / (n1 * c1) * ts ** 2)
        close(res.Tp, n2 * c2 / (n1 * c1) * tp ** 2)

    def test_brewster(self):
        res = Stack(ci(1.0), [], ci(1.5))(wl(), math.atan(1.5))
        close(res.Rp, 0.0, 1e-14)

    def test_total_internal_reflection(self):
        res = Stack(ci(1.5), [], ci(1.0))(wl(), math.radians(60))
        close(res.Rs, 1.0)
        close(res.Rp, 1.0)
        close(res.Ts, 0.0)
        close(res.Tp, 0.0)

    def test_absorbing_substrate_ellipsometry(self):
        N = complex(3.9, 0.02)
        th = math.radians(70)
        s, c = math.sin(th), math.cos(th)
        c2 = (1 - (s / N) ** 2) ** 0.5
        rs = (c - N * c2) / (c + N * c2)
        rp = (N * c - c2) / (N * c + c2)
        res = Stack(ci(1.0), [], ci(N.real, N.imag))(wl(), th)
        rho = rp / rs
        close(res.psi, math.atan(abs(rho)))
        close(res.delta, math.atan2(rho.imag, rho.real))


# ── coherent thin films ────────────────────────────────────────────────────

class TestCoherent:
    def test_quarter_wave_ar(self):
        ns = 2.25
        nc = math.sqrt(ns)
        res = Stack(ci(1.0), [(ci(nc), WL0 / (4 * nc))], ci(ns))(wl())
        close(res.R, 0.0, 1e-14)

    def test_half_wave_is_absentee(self):
        bare = Stack(ci(1.0), [], ci(1.5))(wl())
        film = Stack(ci(1.0), [(ci(2.0), WL0 / (2 * 2.0))], ci(1.5))(wl())
        close(film.R, bare.R)

    def test_bragg_mirror(self):
        n0, nH, nL, ns, N = 1.0, 2.3, 1.45, 1.52, 6
        layers = [(ci(nH), WL0 / (4 * nH)), (ci(nL), WL0 / (4 * nL))] * N
        res = Stack(ci(n0), layers, ci(ns))(wl())
        a, b = n0 * nL ** (2 * N), ns * nH ** (2 * N)
        close(res.R, ((a - b) / (a + b)) ** 2)

    def test_zero_thickness_layer_is_transparent(self):
        th = torch.tensor(0.6, dtype=F64)
        bare = Stack(ci(1.0), [(ci(1.8), 50.0)], ci(1.5))(wl(500, 700), th)
        extra = Stack(ci(1.0), [(ci(1.8), 50.0), (ci(3.0, 0.5), 0.0)], ci(1.5))(wl(500, 700), th)
        close(extra.rs, bare.rs)
        close(extra.rp, bare.rp)

    @pytest.mark.parametrize("theta_deg", [0, 35, 75])
    def test_energy_conservation_lossless(self, theta_deg):
        g = torch.Generator().manual_seed(0)
        ns = 1.3 + 1.2 * torch.rand(7, generator=g, dtype=F64)
        ds = 20 + 300 * torch.rand(7, generator=g, dtype=F64)
        layers = [(ci(float(n)), float(d)) for n, d in zip(ns, ds)]
        res = Stack(ci(1.0), layers, ci(1.6))(torch.linspace(400, 900, 51, dtype=F64),
                                             math.radians(theta_deg))
        close(res.Rs + res.Ts, torch.ones_like(res.Rs))
        close(res.Rp + res.Tp, torch.ones_like(res.Rp))

    def test_absorption_nonnegative(self):
        layers = [(ci(1.5), 80.0), (ci(0.2, 3.5), 15.0), (ci(2.0, 0.1), 200.0)]
        res = Stack(ci(1.0), layers, ci(1.5))(torch.linspace(400, 900, 51, dtype=F64), 0.4)
        assert (res.As > -1e-12).all() and (res.Ap > -1e-12).all()
        assert (res.As < 1).all()

    def test_thick_absorber_is_stable(self):
        metal = ci(0.2, 3.5)
        semi = Stack(ci(1.0), [], metal)(wl(400, 800), 0.3)
        thick = Stack(ci(1.0), [(metal, 1e7)], ci(1.5))(wl(400, 800), 0.3)
        assert torch.isfinite(thick.rs).all() and torch.isfinite(thick.Ts).all()
        close(thick.rs, semi.rs)
        close(thick.Ts, 0.0)

    def test_frustrated_tir_tunnelling(self):
        """Evanescent gap: transmission decays with gap thickness, no overflow."""
        th = math.radians(60)
        T = [Stack(ci(1.5), [(ci(1.0), d)], ci(1.5))(wl(), th).Ts.item()
             for d in (50.0, 200.0, 1e5)]
        assert T[0] > T[1] > 0 and T[2] == pytest.approx(0.0, abs=1e-15)


# ── incoherent substrate ───────────────────────────────────────────────────

class TestIncoherent:
    def test_bare_slab_closed_form(self):
        n = 1.5
        R1 = ((n - 1) / (n + 1)) ** 2
        res = Stack(ci(1.0), [], ci(n), substrate_thickness=1e6)(wl())
        close(res.R, 2 * R1 / (1 + R1))
        close(res.T, (1 - R1) / (1 + R1))

    @pytest.mark.parametrize("theta_deg", [0, 45])
    def test_equals_phase_averaged_coherent(self, theta_deg):
        """Incoherent sum == coherent R/T averaged over one period of substrate phase."""
        th = math.radians(theta_deg)
        films = [(ci(2.1), 70.0), (ci(1.46), 120.0)]
        n_sub, d0 = 1.52, 5000.0
        inc = Stack(ci(1.0), films, ci(n_sub), substrate_thickness=d0)(wl(), th)

        kz = math.sqrt(n_sub ** 2 - math.sin(th) ** 2)
        period = WL0 / (2 * kz)
        n_avg = 4000
        ds = d0 + period * torch.arange(n_avg, dtype=F64) / n_avg
        Rs, Rp, Ts, Tp = 0.0, 0.0, 0.0, 0.0
        for d in ds:
            coh = Stack(ci(1.0), films + [(ci(n_sub), float(d))], ci(1.0))(wl(), th)
            Rs, Rp = Rs + coh.Rs, Rp + coh.Rp
            Ts, Tp = Ts + coh.Ts, Tp + coh.Tp
        close(inc.Rs, Rs / n_avg, 1e-9)
        close(inc.Rp, Rp / n_avg, 1e-9)
        close(inc.Ts, Ts / n_avg, 1e-9)
        close(inc.Tp, Tp / n_avg, 1e-9)

    def test_absorbing_substrate_beer_lambert(self):
        n, k, d = 1.5, 1e-3, 2e5
        res = Stack(ci(1.0), [], ci(n, k), substrate_thickness=d)(wl())
        R1 = ((n - 1) / (n + 1)) ** 2            # k is negligible at the interfaces
        a = math.exp(-4 * math.pi * k * d / WL0)
        T_exp = (1 - R1) ** 2 * a / (1 - R1 ** 2 * a ** 2)
        close(res.T, T_exp, 1e-6)
        assert (res.A > 0).all()

    def test_energy_conservation_lossless(self):
        films = [(ci(2.1), 70.0), (ci(1.46), 120.0)]
        res = Stack(ci(1.0), films, ci(1.52), substrate_thickness=1e6,
                    exit_medium=ci(1.33))(torch.linspace(400, 900, 21, dtype=F64), 0.5)
        close(res.Rs + res.Ts, torch.ones_like(res.Rs))
        close(res.Rp + res.Tp, torch.ones_like(res.Rp))

    def test_ellipsometry_unavailable(self):
        res = Stack(ci(1.0), [], ci(1.5), substrate_thickness=1e6)(wl())
        assert res.rs is None
        with pytest.raises(ValueError, match="incoherent"):
            res.psi

    def test_exit_medium_requires_thickness(self):
        with pytest.raises(ValueError, match="substrate_thickness"):
            Stack(ci(1.0), [], ci(1.5), exit_medium=ci(1.0))


# ── module behaviour: broadcasting, gradients, dtype, presets ──────────────

class TestModule:
    def test_broadcast_wl_theta(self):
        stack = Stack(ci(1.0), [(ci(2.0), 100.0)], ci(1.5))
        w = torch.linspace(400, 800, 5, dtype=F64)
        t = torch.linspace(0, 1.2, 3, dtype=F64)
        res = stack(w[:, None], t[None, :])
        assert res.Rs.shape == (5, 3)
        close(res.Rs[:, 1], stack(w, t[1]).Rs)

    def test_thickness_and_model_gradients(self):
        d = nn.Parameter(torch.tensor(90.0, dtype=F64))
        n_film = nn.Parameter(torch.tensor(1.9, dtype=F64))
        d_sub = nn.Parameter(torch.tensor(2e4, dtype=F64))
        film = ConstantIndex(n_film, 0.01, dtype=F64)
        for kw in ({}, {"substrate_thickness": d_sub}):
            stack = Stack(ci(1.0), [Layer(film, d)], ci(1.5, 1e-4), **kw)
            assert {id(p) for p in stack.parameters()} >= {id(d), id(film.nc)}
            stack(torch.linspace(400, 800, 11, dtype=F64), 0.3).R.sum().backward()
            assert d.grad is not None and torch.isfinite(d.grad)
            assert film.nc.grad is not None and torch.isfinite(film.nc.grad)
            d.grad = None
            film.nc.grad = None
        assert d_sub.grad is not None and torch.isfinite(d_sub.grad)

    def test_gradcheck(self):
        def f(d, n):
            film = ConstantIndex(n, 0.05, dtype=F64)
            res = Stack(ci(1.0), [Layer(film, d)], ci(3.5, 0.1))(wl(500, 650), 0.7)
            return torch.stack([res.Rs, res.Rp, res.psi, res.delta])
        d = torch.tensor(120.0, dtype=F64, requires_grad=True)
        n = torch.tensor(1.8, dtype=F64, requires_grad=True)
        assert torch.autograd.gradcheck(f, (d, n))

    def test_float32_default_runs(self):
        stack = Stack(material("air"), [(material("SiO2"), 100.0)], material("Au"))
        res = stack(torch.linspace(400, 800, 9))
        assert res.rs.dtype == torch.complex64
        assert torch.isfinite(res.R).all()

    def test_double_promotes(self):
        stack = Stack(material("air"), [(material("SiO2"), 100.0)], material("Au")).double()
        res = stack(torch.linspace(400, 800, 9, dtype=F64))
        assert res.rs.dtype == torch.complex128
        assert stack.layers[0].d.dtype == F64

    def test_state_dict_roundtrip(self):
        s1 = Stack(ci(1.0), [(ci(2.0), 100.0)], ci(1.5), substrate_thickness=1e6)
        s2 = Stack(ci(1.0), [(ci(2.0), 50.0)], ci(1.5), substrate_thickness=2e6)
        s2.load_state_dict(s1.state_dict())
        close(s2.layers[0].d, 100.0)
        close(s2(wl()).R, s1(wl()).R)
