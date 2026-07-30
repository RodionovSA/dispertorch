"""Tests for bundled material presets (metals + dielectrics/semiconductors)."""

import torch
import pytest

from dispertorch import material, list_materials
from dispertorch.base import Composite
from dispertorch.constant import ConstantIndex

WL_VIS  = torch.linspace(400.0,  800.0, 100)   # visible, nm
WL_WIDE = torch.linspace(300.0, 1200.0, 200)   # wider sweep, nm

METALS = ["Au", "Ag", "Cu", "Al"]

# Wavelength ranges matched to each material's stated validity window
_DIELECTRIC_WL = {
    "SiO2":  torch.linspace(250.0,  1500.0, 100),   # 0.25–1.5 µm
    "Si3N4": torch.linspace(400.0,  2000.0, 100),   # 0.4–2.0 µm
    "cSi":   torch.linspace(1400.0, 5000.0, 100),   # 1.4–5.0 µm (IR only)
    "TiO2":  torch.linspace(450.0,  1200.0, 100),   # 0.45–1.2 µm
}

# ── Metals (existing) ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("metal", METALS)
def test_builds_as_composite(metal):
    """material() must return a Composite without raising."""
    assert isinstance(material(metal), Composite)


@pytest.mark.parametrize("metal", METALS)
def test_eps2_nonneg(metal):
    """eps'' ≥ 0 (no gain) across a wide wavelength sweep."""
    eps = material(metal).permittivity(WL_WIDE)
    assert (eps.imag >= -1e-5).all(), f"{metal}: eps'' < 0"


@pytest.mark.parametrize("metal", METALS)
def test_k_nonneg(metal):
    """Extinction coefficient ≥ 0 (principal-root convention)."""
    assert (material(metal).k(WL_WIDE) >= -1e-5).all(), f"{metal}: k < 0"


@pytest.mark.parametrize("metal", METALS)
def test_metallic_eps1_negative(metal):
    """All four metals must have eps' < 0 somewhere in the visible."""
    eps = material(metal).permittivity(WL_VIS)
    assert (eps.real < 0).any(), f"{metal}: eps' never negative in visible"


def test_au_600nm_sanity():
    """Gold at 600 nm: n < 1 and k > 2 (literature: n≈0.3, k≈3.0)."""
    m = material("Au")
    wl = torch.tensor([600.0])
    n, k = m.nk(wl)
    assert n.item() < 1.0, f"Au 600 nm: n={n.item():.3f} — expected < 1"
    assert k.item() > 2.0, f"Au 600 nm: k={k.item():.3f} — expected > 2"


def test_explicit_source_matches_default():
    """material('Au', source='rakic-ld') must equal the default source."""
    wl = torch.linspace(400.0, 800.0, 50)
    eps_default  = material("Au").permittivity(wl)
    eps_explicit = material("Au", source="rakic-ld").permittivity(wl)
    torch.testing.assert_close(eps_default, eps_explicit)


# ── Lossless dielectrics ──────────────────────────────────────────────────────

@pytest.mark.parametrize("name", list(_DIELECTRIC_WL))
def test_dielectric_builds(name):
    """Each dielectric preset must build without raising."""
    material(name)


@pytest.mark.parametrize("name", list(_DIELECTRIC_WL))
def test_dielectric_lossless(name):
    """Sellmeier-based dielectrics must have eps'' ≈ 0 (lossless)."""
    wl = _DIELECTRIC_WL[name]
    eps = material(name).permittivity(wl)
    assert (eps.imag.abs() < 1e-5).all(), f"{name}: eps'' not zero"


@pytest.mark.parametrize("name", list(_DIELECTRIC_WL))
def test_dielectric_k_zero(name):
    """Extinction coefficient must be 0 in the validity window."""
    wl = _DIELECTRIC_WL[name]
    assert (material(name).k(wl).abs() < 1e-5).all(), f"{name}: k != 0"


@pytest.mark.parametrize("name", list(_DIELECTRIC_WL))
def test_dielectric_n_finite_positive(name):
    """Refractive index must be finite and > 1 throughout the window."""
    wl = _DIELECTRIC_WL[name]
    n = material(name).n(wl)
    assert torch.isfinite(n).all(), f"{name}: n not finite"
    assert (n > 1.0).all(), f"{name}: n ≤ 1.0 somewhere"


# ── Reference-index assertions (atol 1e-3) ───────────────────────────────────

def test_sio2_malitson_589nm():
    """SiO2 (Malitson) @ 587.6 nm → n ≈ 1.4585."""
    n = material("SiO2").n(torch.tensor([587.6])).item()
    assert abs(n - 1.4585) < 1e-3, f"SiO2: n={n:.6f}, expected ≈ 1.4585"


def test_si3n4_luke_1550nm():
    """Si3N4 (Luke 2015) @ 1550 nm → n ≈ 1.996."""
    n = material("Si3N4").n(torch.tensor([1550.0])).item()
    assert abs(n - 1.996) < 1e-3, f"Si3N4: n={n:.6f}, expected ≈ 1.996"


def test_csi_salzberg_1550nm():
    """cSi (Salzberg & Villa) @ 1550 nm → n ≈ 3.478."""
    n = material("cSi").n(torch.tensor([1550.0])).item()
    assert abs(n - 3.478) < 1e-3, f"cSi: n={n:.6f}, expected ≈ 3.478"


def test_tio2_devore_633nm():
    """TiO2 (Devore, rutile o-ray) @ 633 nm → n ≈ 2.584."""
    n = material("TiO2").n(torch.tensor([633.0])).item()
    assert abs(n - 2.584) < 1e-3, f"TiO2: n={n:.6f}, expected ≈ 2.584"


def test_sio2_explicit_source():
    """material('SiO2', source='malitson') must equal the default."""
    wl = torch.linspace(400.0, 1200.0, 50)
    eps_default  = material("SiO2").permittivity(wl)
    eps_explicit = material("SiO2", source="malitson").permittivity(wl)
    torch.testing.assert_close(eps_default, eps_explicit)


# ── Amorphous silicon ─────────────────────────────────────────────────────────

def test_asi_builds_as_composite():
    assert isinstance(material("aSi"), Composite)


def test_asi_absorbing_visible():
    """aSi must have k > 0 somewhere in the visible (it's a semiconductor absorber)."""
    k = material("aSi").k(WL_VIS)
    assert (k > 0).any(), "aSi: k never positive in visible"


def test_asi_eps2_nonneg():
    """aSi must satisfy eps'' ≥ 0 (no gain) across a broad sweep."""
    wl = torch.linspace(250.0, 1000.0, 200)
    eps = material("aSi").permittivity(wl)
    assert (eps.imag >= -1e-5).all(), "aSi: eps'' < 0"


def test_asi_finite():
    """aSi n and k must be finite."""
    wl = torch.linspace(300.0, 900.0, 100)
    n, k = material("aSi").nk(wl)
    assert torch.isfinite(n).all() and torch.isfinite(k).all()


# ── Air ────────────────────────────────────────────────────────────────────

def test_air_builds_as_constant_index():
    """material('air') must return a ConstantIndex."""
    assert isinstance(material("air"), ConstantIndex)


def test_air_is_unity_lossless():
    """Air (unity source) must have n == 1 and k == 0 everywhere."""
    m = material("air")
    n, k = m.nk(WL_WIDE)
    torch.testing.assert_close(n, torch.ones_like(WL_WIDE))
    torch.testing.assert_close(k, torch.zeros_like(WL_WIDE))
    eps = m.permittivity(WL_WIDE)
    torch.testing.assert_close(eps, torch.ones_like(WL_WIDE, dtype=torch.complex64))


# ── list_materials ────────────────────────────────────────────────────────────

def test_list_materials_contains_all():
    lm = list_materials()
    for name in ["air", "Au", "Ag", "Cu", "Al", "SiO2", "Si3N4", "cSi", "TiO2", "aSi"]:
        assert name in lm, f"{name} missing from list_materials()"


def test_list_materials_source_keys():
    lm = list_materials()
    assert lm["air"]  == ["unity"]
    assert lm["SiO2"] == ["malitson"]
    assert lm["cSi"]  == ["salzberg"]
    assert lm["aSi"]  == ["jellison-modine"]
