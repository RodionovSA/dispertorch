"""Tests for Cauchy and Sellmeier transparent-material dispersion models."""

import pytest
import torch
import torch.nn as nn

from dispertorch import Cauchy, Sellmeier
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


# ── Registry ───────────────────────────────────────────────────────────────

def test_registry():
    assert "Cauchy" in MODEL_REGISTRY
    assert "Sellmeier" in MODEL_REGISTRY
