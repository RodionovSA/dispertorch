"""Tests for ConstantIndex and ConstantEps (non-dispersive terms)."""

import torch
import torch.nn as nn
import pytest

from dispertorch import ConstantEps, ConstantIndex, DispersionModel
from dispertorch.base import Composite
from dispertorch.utils import register

WL = torch.linspace(400.0, 800.0, 50)


# ── ConstantIndex ──────────────────────────────────────────────────────────

class TestConstantIndex:
    def test_constructs_and_matches_index(self):
        """Regression test: ConstantIndex must construct without the n/k
        buffer-name collision, and n(wl)/k(wl) must reflect the stored value."""
        m = ConstantIndex(1.5)
        torch.testing.assert_close(m.n(WL), torch.full_like(WL, 1.5))
        torch.testing.assert_close(m.k(WL), torch.zeros_like(WL))
        eps = m.permittivity(WL)
        torch.testing.assert_close(eps, torch.full_like(WL, 2.25, dtype=torch.complex64))

    def test_absorbing_case(self):
        """Pass k for an absorbing constant material; eps'' >= 0 and nk() round-trips."""
        m = ConstantIndex(1.5, 0.1)
        n, k = m.nk(WL)
        torch.testing.assert_close(n, torch.full_like(WL, 1.5), atol=1e-5, rtol=0)
        torch.testing.assert_close(k, torch.full_like(WL, 0.1), atol=1e-5, rtol=0)
        eps = m.permittivity(WL)
        assert (eps.imag >= -1e-5).all(), "ConstantIndex: eps'' < 0"

    def test_stored_names_are_nc_kc(self):
        """Stored constants live under nc/kc (state_dict keys); n/k stay methods."""
        m = ConstantIndex(1.5, 0.1)
        buf_names = {name for name, _ in m.named_buffers()}
        assert {"nc", "kc"} <= buf_names
        assert set(m.state_dict().keys()) == {"nc", "kc"}
        assert callable(m.n) and callable(m.k)

    def test_gradient_flows(self):
        """Wrapping n in nn.Parameter must make it an optimizable gradient leaf."""
        m = ConstantIndex(nn.Parameter(torch.tensor(1.5)))
        assert len(list(m.parameters())) == 1
        m.n(WL).sum().backward()
        assert m.nc.grad is not None
        assert torch.isfinite(m.nc.grad).all()

    def test_permittivity_broadcasts_to_wl_shape(self):
        m = ConstantIndex(1.5)
        wl = torch.linspace(400.0, 800.0, 7)
        assert m.permittivity(wl).shape == wl.shape

    def test_composes_with_constant_eps(self):
        """ConstantEps + ConstantIndex must build a Composite and sum permittivities."""
        eps_bg = ConstantEps(1.0)
        idx = ConstantIndex(1.5)
        comp = eps_bg + idx
        assert isinstance(comp, Composite)
        torch.testing.assert_close(
            comp.permittivity(WL),
            eps_bg.permittivity(WL) + idx.permittivity(WL),
        )


# ── ConstantEps ────────────────────────────────────────────────────────────

class TestConstantEps:
    def test_default_is_real(self):
        m = ConstantEps(2.25)
        eps = m.permittivity(WL)
        torch.testing.assert_close(eps, torch.full_like(WL, 2.25, dtype=torch.complex64))

    def test_complex_case(self):
        m = ConstantEps(2.25, 0.1)
        eps = m.permittivity(WL)
        assert (eps.imag >= -1e-5).all()
        torch.testing.assert_close(eps.imag, torch.full_like(WL, 0.1))


# ── register() reserved-name guard ────────────────────────────────────────

def test_register_rejects_reserved_name():
    """register() must raise a clear error instead of the opaque nn.Module
    KeyError when a coefficient name shadows a DispersionModel accessor."""

    class Bad(DispersionModel):
        def __init__(self):
            super().__init__()
            register(self, "n", 1.0)

        def permittivity(self, wl):
            return torch.ones_like(wl) + 0j

    with pytest.raises(ValueError, match="shadows"):
        Bad()


def test_register_allows_ordinary_names():
    """Sanity check: the guard must not trip on non-colliding coefficient names."""

    class Fine(DispersionModel):
        def __init__(self):
            super().__init__()
            register(self, "n0", 1.5)

        def permittivity(self, wl):
            return torch.ones_like(wl) + 0j

    Fine()  # must not raise
