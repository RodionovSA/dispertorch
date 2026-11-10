"""Tests for Cauchy and Sellmeier transparent-material dispersion models."""

import pytest
import torch
import torch.nn as nn

from dispertorch import Cauchy, CauchyUrbach, Sellmeier
from dispertorch.utils import MODEL_REGISTRY

WL = torch.linspace(400.0, 800.0, 200)   # visible sweep, nm

# ── Sellmeier reference data ───────────────────────────────────────────────
# SCHOTT N-BK7 (from SCHOTT datasheet)
BK7_B = [1.03961212, 0.231792344, 1.01046945]
BK7_C = [0.00600069867, 0.0200179144, 103.560653]   # µm²

# Fused silica — Malitson 1965; C values pre-squared to µm²
FS_B = [0.6961663, 0.4079426, 0.8974794]
FS_C = [0.0684043**2, 0.1162414**2, 9.896161**2]     # µm²


# ── Cauchy ─────────────────────────────────────────────────────────────────

class TestCauchy:
    def test_lossless_eps_imag_zero(self):
        """Cauchy is a real-index model: eps'' must be exactly 0."""
        m = Cauchy(A=1.49, B=3.54e-3)
        eps = m.permittivity(WL)
        assert (eps.imag.abs() < 1e-7).all(), "Cauchy: eps'' != 0"

    def test_k_zero(self):
        """Extinction coefficient must be 0 everywhere."""
        m = Cauchy(A=1.49, B=3.54e-3)
        assert (m.k(WL).abs() < 1e-6).all()

    def test_normal_dispersion(self):
        """n must decrease monotonically with wavelength (normal dispersion for B > 0)."""
        m = Cauchy(A=1.49, B=3.54e-3)
        n = m.n(WL)
        assert (n.diff() < 0).all(), "Cauchy: not normal dispersion"

    def test_n_in_glass_range(self):
        """Typical glass index 1.4–1.6 at 589 nm."""
        m = Cauchy(A=1.49, B=3.54e-3)
        n589 = m.n(torch.tensor([589.0])).item()
        assert 1.4 < n589 < 1.6, f"Cauchy: n={n589:.4f} out of glass range"

    def test_eps_equals_n_squared(self):
        """eps.real must equal n²."""
        m = Cauchy(A=1.50, B=4.0e-3, C=1.0e-6)
        eps = m.permittivity(WL)
        n   = m.n(WL)
        torch.testing.assert_close(eps.real, n ** 2, atol=1e-5, rtol=0)

    def test_gradient_flows(self):
        """Backprop must reach nn.Parameter coefficients."""
        m = Cauchy(A=nn.Parameter(torch.tensor(1.5)),
                   B=nn.Parameter(torch.tensor(4e-3)))
        m.n(WL).sum().backward()
        assert m.A.grad is not None
        assert m.B.grad is not None
        assert torch.isfinite(m.A.grad).all()
        assert torch.isfinite(m.B.grad).all()

    def test_scalar_args_are_buffers(self):
        m = Cauchy(1.49, 3.54e-3)
        buf_names = {n for n, _ in m.named_buffers()}
        assert {"A", "B", "C"} <= buf_names
        assert len(list(m.parameters())) == 0


# ── Sellmeier ──────────────────────────────────────────────────────────────

class TestSellmeier:
    def test_bk7_at_589nm(self):
        """N-BK7 at 587.6 nm must give n ≈ 1.5168 (SCHOTT value)."""
        m = Sellmeier(BK7_B, BK7_C)
        n = m.n(torch.tensor([587.6])).item()
        assert abs(n - 1.5168) < 1e-3, f"BK7: n={n:.6f}, expected ≈ 1.5168"

    def test_fused_silica_at_589nm(self):
        """Fused silica at 587.6 nm must give n ≈ 1.4585 (Malitson 1965)."""
        m = Sellmeier(FS_B, FS_C)
        n = m.n(torch.tensor([587.6])).item()
        assert abs(n - 1.4585) < 1e-3, f"Fused silica: n={n:.6f}, expected ≈ 1.4585"

    def test_lossless_eps_imag_zero(self):
        """eps'' must be exactly 0 (lossless)."""
        m = Sellmeier(BK7_B, BK7_C)
        eps = m.permittivity(WL)
        assert (eps.imag.abs() < 1e-6).all(), "Sellmeier: eps'' != 0"

    def test_k_zero(self):
        m = Sellmeier(BK7_B, BK7_C)
        assert (m.k(WL).abs() < 1e-6).all()

    def test_normal_dispersion(self):
        """BK7 n must decrease monotonically across the visible."""
        m = Sellmeier(BK7_B, BK7_C)
        n = m.n(WL)
        assert (n.diff() < 0).all(), "Sellmeier BK7: not normal dispersion"

    def test_eps_equals_n_squared(self):
        m = Sellmeier(BK7_B, BK7_C)
        eps = m.permittivity(WL)
        n   = m.n(WL)
        torch.testing.assert_close(eps.real, n ** 2, atol=1e-5, rtol=0)

    def test_gradient_flows_scalar_C(self):
        """Gradient must reach a 1-D nn.Parameter B."""
        m = Sellmeier(
            nn.Parameter(torch.tensor([1.04, 0.23, 1.01])),
            [0.006, 0.020, 103.6],
        )
        m.n(WL).sum().backward()
        assert m.B.grad is not None
        assert torch.isfinite(m.B.grad).all()

    def test_mismatched_lengths_raises(self):
        """B and C with different lengths must raise ValueError."""
        with pytest.raises(ValueError, match="same shape"):
            Sellmeier([1.0, 2.0], [0.01])

    def test_scalar_args_are_buffers(self):
        m = Sellmeier(BK7_B, BK7_C)
        buf_names = {n for n, _ in m.named_buffers()}
        assert {"B", "C"} <= buf_names
        assert len(list(m.parameters())) == 0

    def test_single_term(self):
        """Single-term Sellmeier (length-1 B, C) must not raise and give finite n."""
        m = Sellmeier([1.0], [0.01])
        n = m.n(WL)
        assert torch.isfinite(n).all()


# ── Cauchy-Urbach ─────────────────────────────────────────────────────────

HC = 1239.841984  # eV·nm


class TestCauchyUrbach:
    def test_zero_amp_reduces_to_cauchy(self):
        """With k_amp = 0 the model must equal plain Cauchy."""
        cu = CauchyUrbach(A=1.50, B=4.0e-3, C=1.0e-6, k_exp=1.5)
        c = Cauchy(A=1.50, B=4.0e-3, C=1.0e-6)
        torch.testing.assert_close(cu.permittivity(WL), c.permittivity(WL))
        assert (cu.k(WL).abs() < 1e-6).all()

    def test_n_and_k_are_exact(self):
        """n is exactly Cauchy and k is exactly the Urbach exponential."""
        wl = WL.double()
        A, B, C, ka, ke, Eb = 1.6, 5e-3, 2e-5, 0.02, 1.3, 3.2
        m = CauchyUrbach(A, B, C, k_amp=ka, k_exp=ke, E_band=Eb, dtype=torch.float64)
        lam = wl * 1e-3
        n_ref = A + B / lam**2 + C / lam**4
        k_ref = ka * torch.exp(ke * (HC / wl - Eb))
        n, k = m.nk(wl)
        torch.testing.assert_close(n, n_ref)
        torch.testing.assert_close(k, k_ref)

    def test_k_equals_amp_at_band_edge(self):
        m = CauchyUrbach(A=1.5, B=4e-3, k_amp=0.01, k_exp=2.0, E_band=3.0,
                         dtype=torch.float64)
        k = m.k(torch.tensor([HC / 3.0], dtype=torch.float64))
        torch.testing.assert_close(k, torch.tensor([0.01], dtype=torch.float64))

    def test_default_band_edge_is_400nm(self):
        m = CauchyUrbach(A=1.5, B=4e-3, k_amp=0.01, k_exp=2.0, dtype=torch.float64)
        k = m.k(torch.tensor([400.0], dtype=torch.float64))
        torch.testing.assert_close(k, torch.tensor([0.01], dtype=torch.float64))

    def test_absorption_decays_with_wavelength(self):
        """For k_exp > 0, k increases monotonically towards short wavelengths."""
        m = CauchyUrbach(A=1.5, B=4e-3, k_amp=0.01, k_exp=1.5)
        k = m.k(WL)
        assert (k > 0).all()
        assert (k.diff() < 0).all()

    def test_absorbing_sign_convention(self):
        """eps'' >= 0 for absorption (library-wide convention)."""
        m = CauchyUrbach(A=1.5, B=4e-3, k_amp=0.05, k_exp=1.5)
        assert (m.permittivity(WL).imag > 0).all()

    def test_gradient_flows_to_all_params(self):
        names = ("A", "B", "C", "k_amp", "k_exp")
        vals = (1.5, 4e-3, 1e-5, 0.01, 1.5)
        m = CauchyUrbach(*(nn.Parameter(torch.tensor(v)) for v in vals))
        n, k = m.nk(WL)
        (n.sum() + k.sum()).backward()
        for name in names:
            g = getattr(m, name).grad
            assert g is not None and torch.isfinite(g).all() and g.abs() > 0, name

    def test_scalar_args_are_buffers(self):
        m = CauchyUrbach(A=1.5, B=4e-3, k_amp=0.01)
        assert len(list(m.parameters())) == 0
        assert {"A", "B", "C", "k_amp", "k_exp", "E_band"} <= set(m.state_dict())


# ── Registry ───────────────────────────────────────────────────────────────

def test_registry():
    assert "Cauchy" in MODEL_REGISTRY
    assert "Sellmeier" in MODEL_REGISTRY
    assert MODEL_REGISTRY["CauchyUrbach"] is CauchyUrbach
