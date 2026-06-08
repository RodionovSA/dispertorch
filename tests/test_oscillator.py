"""Tests for Drude, Lorentz, and TaucLorentz oscillator models."""

import torch
import torch.nn as nn
import pytest

from dispertorch import ConstantEps, Drude, Lorentz, TaucLorentz
from dispertorch.oscillator import _HC_EV_NM
from dispertorch.utils import MODEL_REGISTRY

# Dense wavelength sweep covering visible + near-IR (nm)
WL = torch.linspace(300.0, 1200.0, 300)

# Representative Tauc-Lorentz parameters (a-Si:H-like)
_TL = dict(A=122.0, omega_0=3.64, gamma=1.96, E_g=1.6)


# ── Drude ──────────────────────────────────────────────────────────────────

class TestDrude:
    def test_eps2_nonneg(self):
        """eps'' must be ≥ 0 (no gain)."""
        m = Drude(15.0, 0.1)
        eps = m.permittivity(WL)
        assert (eps.imag >= -1e-6).all(), "Drude: eps'' < 0"

    def test_eps1_negative(self):
        """Free electrons give eps' < 0 for all positive energies."""
        m = Drude(15.0, 0.1)
        eps = m.permittivity(WL)
        assert (eps.real < 0).all(), "Drude: eps' must be negative"

    def test_k_nonneg(self):
        """Extinction coefficient must be ≥ 0 (principal sqrt convention)."""
        m = Drude(15.0, 0.1)
        assert (m.k(WL) >= -1e-6).all()

    def test_gradient_flows(self):
        """Backprop through permittivity must reach nn.Parameter coefficients."""
        m = Drude(nn.Parameter(torch.tensor(15.0)), 0.1)
        eps = m.permittivity(WL)
        eps.real.sum().backward()
        assert m.omega_p.grad is not None

    def test_scalar_args_are_buffers(self):
        """Plain scalar args must be stored as non-gradient buffers."""
        m = Drude(15.0, 0.1)
        buf_names = {n for n, _ in m.named_buffers()}
        assert {"omega_p", "gamma"} <= buf_names
        assert len(list(m.parameters())) == 0


# ── Lorentz ────────────────────────────────────────────────────────────────

class TestLorentz:
    def test_eps2_nonneg(self):
        """eps'' must be ≥ 0 for positive A and gamma."""
        m = Lorentz(1.0, 3.0, 0.2)
        eps = m.permittivity(WL)
        assert (eps.imag >= -1e-6).all(), "Lorentz: eps'' < 0"

    def test_eps2_peaks_near_resonance(self):
        """eps'' should be largest close to the resonance wavelength."""
        omega_0 = 3.0          # eV
        m = Lorentz(1.0, omega_0, 0.1)
        eps = m.permittivity(WL)
        wl_resonance = _HC_EV_NM / omega_0   # ≈ 413 nm
        peak_wl = WL[eps.imag.argmax()].item()
        assert abs(peak_wl - wl_resonance) < 30.0, (
            f"Lorentz peak at {peak_wl:.1f} nm, expected ≈ {wl_resonance:.1f} nm"
        )

    def test_k_nonneg(self):
        m = Lorentz(1.0, 3.0, 0.2)
        assert (m.k(WL) >= -1e-6).all()

    def test_gradient_flows(self):
        m = Lorentz(nn.Parameter(torch.tensor(1.0)), 3.0, 0.2)
        eps = m.permittivity(WL)
        eps.real.sum().backward()
        assert m.A.grad is not None

    def test_composition_shifts_eps1(self):
        """ConstantEps(c) + Lorentz must shift eps' by exactly c."""
        eps_inf_val = 2.0
        m_plain = Lorentz(1.0, 3.0, 0.2)
        m_comp = ConstantEps(eps_inf_val) + m_plain
        eps_plain = m_plain.permittivity(WL)
        eps_comp = m_comp.permittivity(WL)
        torch.testing.assert_close(
            eps_comp.real - eps_plain.real,
            torch.full_like(eps_plain.real, eps_inf_val),
            atol=1e-5, rtol=0,
        )

    def test_scalar_args_are_buffers(self):
        m = Lorentz(1.0, 3.0, 0.2)
        buf_names = {n for n, _ in m.named_buffers()}
        assert {"A", "omega_0", "gamma"} <= buf_names
        assert len(list(m.parameters())) == 0


# ── TaucLorentz ────────────────────────────────────────────────────────────

class TestTaucLorentz:
    def test_eps2_zero_below_gap(self):
        """eps_2 must be exactly 0 for all E ≤ E_g (λ > λ_gap)."""
        m = TaucLorentz(**_TL)
        # Wavelengths well above the gap wavelength → E < E_g
        wl_long = torch.linspace(_HC_EV_NM / _TL["E_g"] + 20, 2500.0, 200)
        eps = m.permittivity(wl_long)
        assert (eps.imag.abs() < 1e-6).all(), "TaucLorentz: eps_2 nonzero below gap"

    def test_eps2_nonneg(self):
        """eps_2 must be ≥ 0 everywhere (no gain)."""
        m = TaucLorentz(**_TL)
        wl = torch.linspace(200.0, 1000.0, 400)
        eps = m.permittivity(wl)
        assert (eps.imag >= -1e-6).all(), "TaucLorentz: eps_2 < 0"

    def test_eps1_finite(self):
        """eps_1 must be finite at all wavelengths (no inf/nan)."""
        m = TaucLorentz(**_TL)
        eps = m.permittivity(WL)
        assert torch.isfinite(eps.real).all(), "TaucLorentz: non-finite eps_1"

    def test_k_nonneg(self):
        m = TaucLorentz(**_TL)
        assert (m.k(WL) >= -1e-6).all()

    def test_gradient_flows_all_params(self):
        """Gradients must reach all four nn.Parameter coefficients."""
        m = TaucLorentz(
            nn.Parameter(torch.tensor(122.0)),
            nn.Parameter(torch.tensor(3.64)),
            nn.Parameter(torch.tensor(1.96)),
            nn.Parameter(torch.tensor(1.6)),
        )
        wl = torch.linspace(300.0, 800.0, 100)
        eps = m.permittivity(wl)
        # Sum both real and imaginary to exercise paths through eps_1 and eps_2
        (eps.real.sum() + eps.imag.sum()).backward()
        for name in ("A", "omega_0", "gamma", "E_g"):
            p = getattr(m, name)
            assert p.grad is not None, f"TaucLorentz: no grad for {name}"
            assert torch.isfinite(p.grad).all(), f"TaucLorentz: non-finite grad for {name}"

    def test_scalar_args_are_buffers(self):
        m = TaucLorentz(**_TL)
        buf_names = {n for n, _ in m.named_buffers()}
        assert {"A", "omega_0", "gamma", "E_g"} <= buf_names
        assert len(list(m.parameters())) == 0


# ── Registry ───────────────────────────────────────────────────────────────

def test_registry_contains_new_models():
    assert "Drude" in MODEL_REGISTRY
    assert "Lorentz" in MODEL_REGISTRY
    assert "TaucLorentz" in MODEL_REGISTRY
