# dispertorch

Differentiable refractive-index dispersion models built on `torch.nn.Module`.

**Permittivity is the primitive.** Every model implements `permittivity(wl)` returning
complex relative permittivity ε = ε′ + iε″ (ε″ ≥ 0 for absorbing media, `exp(-iωt)`
convention). Refractive index n, k, and the complex index N = n + ik are derived.
Models compose with `+`, are differentiable end-to-end via autograd, and move with
`.to(device/dtype)`.

## Install

**End users** — install directly from GitHub, no clone needed:

```bash
pip install git+https://github.com/RodionovSA/dispertorch.git
```

Only `torch>=2.4` is pulled in; plotting and testing tools are not included.

**Developers** — clone and install in editable mode with dev extras:

```bash
git clone https://github.com/RodionovSA/dispertorch.git
cd dispertorch

# with pip
pip install -e ".[dev]"
pytest          # 36 tests

# with uv
uv sync --extra dev
uv run pytest
```

## Quick start

```python
import torch
from dispertorch import material

au = material("Au")                       # Rakić 1998 Lorentz-Drude preset
wl = torch.linspace(300, 900, 200)        # wavelengths in nm

n, k   = au.nk(wl)                        # real & imaginary parts; shape (200,)
eps    = au.permittivity(wl)              # complex64; eps.imag >= 0
```

## Core concepts

| Concept | Details |
|---|---|
| **Primitive** | `permittivity(wl: Tensor) → Tensor` (complex; wavelength in **nm**) |
| **Derived** | `n`, `k`, `nk`, `index` — implemented once on `DispersionModel`, not overridden |
| **Units** | Wavelength: **nm**; oscillator energies: **eV** (`E = 1239.84 / λ`) |
| **Sign convention** | ε = ε′ + iε″, ε″ ≥ 0 (`exp(−iωt)`, positive absorption) |
| **Composition** | `a + b` → `Composite`; `sum([a, b, c])` works too (flattens nested composites) |
| **Coefficients** | `nn.Parameter` → optimizable gradient leaf; plain scalar → non-gradient buffer (still moves with `.to()`, saved in `state_dict()`) |

## Model catalog

| Class | Parameters | ε(E) |
|---|---|---|
| `ConstantIndex(n, k=0)` | n, k (dimensionless) | (n + ik)² |
| `ConstantEps(eps_re, eps_im=0)` | ε_re, ε_im (dimensionless) | ε_re + iε_im |
| `Drude(omega_p, gamma)` | ωp [eV], γ [eV] | −ωp² / (E² + iγE) |
| `Lorentz(A, omega_0, gamma)` | A [eV²], ω₀ [eV], γ [eV] | A / (ω₀² − E² − iγE) |
| `TaucLorentz(A, omega_0, gamma, E_g)` | A [eV], ω₀ [eV], γ [eV], E_g [eV] | ε₁ + iε₂ (KK, ε₂=0 below E_g) |
| `Cauchy(A, B, C=0)` | A (dim-less), B [µm²], C [µm⁴] | (A + B/λ² + C/λ⁴)², lossless |
| `Sellmeier(B, C)` | B (dim-less, len K), C [µm²] | 1 + Σ Bᵢλ²/(λ²−Cᵢ), lossless |

Oscillator models return **only their own term** — compose with `ConstantEps(eps_inf)`
for the high-frequency background.
`Cauchy` and `Sellmeier` are lossless (k = 0); coefficients use standard µm units,
wavelength input is still nm.

## Material presets

```python
from dispertorch import material, list_materials

print(list_materials())
# {'Au': ['rakic-ld'], 'Ag': ['rakic-ld'], 'Cu': ['rakic-ld'], 'Al': ['rakic-ld'],
#  'SiO2': ['malitson'], 'Si3N4': ['luke'], 'cSi': ['salzberg'],
#  'TiO2': ['devore-o'], 'aSi': ['jellison-modine']}

au   = material("Au")                      # Rakić 1998 LD preset
sio2 = material("SiO2")                   # Malitson 1965 Sellmeier
si3n4 = material("Si3N4", source="luke")  # Luke 2015 Sellmeier
```

| Material | Source | Model | Notes |
|---|---|---|---|
| `"Au"`, `"Ag"`, `"Cu"`, `"Al"` | Rakić 1998 | `ConstantEps + Drude + Lorentz×N` | metals, 0.2–12.4 µm |
| `"SiO2"` | Malitson 1965 | `Sellmeier` (3-term) | fused silica, 0.21–6.7 µm |
| `"Si3N4"` | Luke 2015 | `Sellmeier` (2-term) | LPCVD, 0.31–5.5 µm |
| `"cSi"` | Salzberg & Villa 1957 | `Sellmeier` (3-term) | **IR only** 1.36–11 µm; lossless |
| `"TiO2"` | Devore 1951 | `ConstantEps + Sellmeier` | rutile ordinary ray, 0.43–1.53 µm |
| `"aSi"` | Jellison & Modine 1996 | `ConstantEps + TaucLorentz` | representative fit; params are sample-dependent |

## Gradient-based fitting

Wrap any coefficient in `nn.Parameter` to make it optimizable:

```python
import torch, torch.nn as nn
from dispertorch import ConstantEps, Lorentz

model = ConstantEps(nn.Parameter(torch.tensor(1.5))) + Lorentz(
    A=nn.Parameter(torch.tensor(2.0)),
    omega_0=nn.Parameter(torch.tensor(2.5)),
    gamma=nn.Parameter(torch.tensor(0.3)),
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
for _ in range(500):
    optimizer.zero_grad()
    n, k = model.nk(wl_fit)
    loss = ((n - n_target)**2 + (k - k_target)**2).mean()
    loss.backward()
    optimizer.step()
```

See **`example.ipynb`** for a full demo with plots, loss curves, and comparison to
ground truth. The notebook also covers single-oscillator sweeps, the preset gallery,
composition patterns, and `state_dict` persistence.


