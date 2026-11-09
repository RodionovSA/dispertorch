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
pytest

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
| `ConstantIndex(n, k=0)` | n, k (dimensionless); stored as `n0`/`k0` | (n + ik)² |
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
# {'air': ['unity'], 'Au': ['rakic-ld'], 'Ag': ['rakic-ld'], 'Cu': ['rakic-ld'], 'Al': ['rakic-ld'],
#  'SiO2': ['malitson'], 'Si3N4': ['luke'], 'cSi': ['salzberg'],
#  'TiO2': ['devore-o'], 'aSi': ['jellison-modine']}

air  = material("air")                     # n = 1 exactly
au   = material("Au")                      # Rakić 1998 LD preset
sio2 = material("SiO2")                   # Malitson 1965 Sellmeier
si3n4 = material("Si3N4", source="luke")  # Luke 2015 Sellmeier
```

| Material | Source | Model | Notes |
|---|---|---|---|
| `"air"` | — | `ConstantIndex` | n = 1 exactly (ambient/vacuum) |
| `"Au"`, `"Ag"`, `"Cu"`, `"Al"` | Rakić 1998 | `ConstantEps + Drude + Lorentz×N` | metals, 0.2–12.4 µm |
| `"SiO2"` | Malitson 1965 | `Sellmeier` (3-term) | fused silica, 0.21–6.7 µm |
| `"Si3N4"` | Luke 2015 | `Sellmeier` (2-term) | LPCVD, 0.31–5.5 µm |
| `"cSi"` | Salzberg & Villa 1957 | `Sellmeier` (3-term) | **IR only** 1.36–11 µm; lossless |
| `"TiO2"` | Devore 1951 | `ConstantEps + Sellmeier` | rutile ordinary ray, 0.43–1.53 µm |
| `"aSi"` | Jellison & Modine 1996 | `ConstantEps + TaucLorentz` | representative fit; params are sample-dependent |

## Planar multilayers

`Stack` computes reflection/transmission of an isotropic planar stack
`ambient | films… | substrate` with recursive Fresnel (Airy) coefficients — stable for
thick absorbing and evanescent layers, differentiable w.r.t. both model coefficients
and thicknesses.

```python
import torch, torch.nn as nn
from dispertorch import Stack, Layer, material

wl    = torch.linspace(400, 900, 300, dtype=torch.float64)   # nm
theta = torch.deg2rad(torch.tensor(70.0, dtype=torch.float64))

stack = Stack(
    material("air"),
    [Layer(material("Si3N4"), nn.Parameter(torch.tensor(80.0))),  # fittable thickness, nm
     (material("SiO2"), 300.0)],                                  # (model, d_nm) also works
    material("aSi"),
).double()

res = stack(wl, theta)
res.Rs, res.Rp, res.Ts, res.Tp        # power R/T per polarization
res.R, res.T, res.A                   # unpolarized
res.rs, res.rp, res.ts, res.tp        # complex amplitudes (E-field)
res.psi, res.delta                    # ellipsometry, radians: rp/rs = tan(psi) e^{i delta}
```

`wl` and `theta` broadcast: `stack(wl[:, None], theta[None, :])` returns `(n_wl, n_theta)` maps.

**Incoherent substrate.** For thick substrates (wafers, glass slides) pass
`substrate_thickness` (nm); multiple reflections inside it add in intensity and light
exits into `exit_medium` (default n = 1):

```python
slide = Stack(material("air"), [(material("TiO2"), 60.0)], material("SiO2"),
              substrate_thickness=1e6)              # 1 mm glass
res = slide(wl)                                       # R, T include backside reflection
```

Amplitudes and `psi`/`delta` are not defined in this mode (the reflected beam is
partially depolarized) and raise `ValueError`.

**Conventions.** Angle in radians in the ambient (which should be transparent).
`r_p = (n₂cosθ₁ − n₁cosθ₂)/(n₂cosθ₁ + n₁cosθ₂)`, so `r_p = −r_s` and Δ = 180° at normal
incidence. With this library's `exp(−iωt)`, n + ik convention, tools that use n − ik
report `conj(ρ)`, i.e. −Δ.

## Fitting

Wrap any coefficient or thickness in `nn.Parameter` to make it free, describe each
dataset as a `Measurement`, and call `fit`. Datasets can mix quantities and angles —
e.g. ellipsometry at 65/70/75° together with normal-incidence R:

```python
import torch, torch.nn as nn
from dispertorch import Stack, Layer, Cauchy, Measurement, fit, material

P = lambda v: nn.Parameter(torch.tensor(v))
stack = Stack(material("air"),
              [Layer(Cauchy(P(1.5), P(0.004)), P(230.0))],   # A, B [µm²], d [nm]
              material("aSi"))

wl  = torch.linspace(400, 1000, 61)           # nm
ang = torch.tensor([65.0, 70.0, 75.0])        # deg

measurements = [
    # variable-angle data: wl[:, None] x theta[None, :] -> data shape (61, 3)
    Measurement("psi",   wl[:, None], psi_data,   ang[None, :], sigma=0.02, degrees=True),
    Measurement("delta", wl[:, None], delta_data, ang[None, :], sigma=0.05, degrees=True),
    Measurement("R",     wl,          R_data,     0.0,          sigma=1e-3),
]

res = fit(stack, measurements,
          method="lm",                 # "lm" (default) | "lbfgs" | "adam"
          max_iter=200, tol=1e-10,
          bounds={"layers.0.d": (0, None)})
print(res.summary())                   # values, standard errors, reduced chi²
```

| Option | Meaning |
|---|---|
| `kind` | Stack: `R Rs Rp T Ts Tp A As Ap psi delta`; DispersionModel: `n k eps1 eps2`; or a callable `f(system, wl, theta)` |
| `sigma` | per-point uncertainty in data units — sets relative weights between datasets and makes `stderr` meaningful |
| `degrees` | `theta`, `psi`, `delta` in degrees (default radians); `delta` residuals are wrapped to ±180° |
| `params` | subset of parameters to fit (objects or names like `"layers.0.d"`); default: all with `requires_grad` |
| `bounds` | `{param_or_name: (low, high)}`, enforced by projection |
| `dtype` | working precision: float64 by default (float32 on Apple MPS, which has no float64), regardless of the system's dtype |
| `lr` | Adam only, as a *relative* step (`1e-2` ≈ 1 % per step) |

**Devices.** The fit runs on the device of the system's parameters (`stack.to("cuda")`,
`.to("mps")`); measurement arrays can live anywhere and are moved there. Parameters
are written back on their own device/dtype; `FitResult` summaries are returned on CPU.

Parameters are updated **in place**. `FitResult` holds `params`, `stderr`, `correlation`,
`reduced_chi2`, per-measurement `residuals` and the χ² `history`.

**Which method?** Levenberg–Marquardt (`"lm"`, Jacobian via `torch.func.jacfwd`) is the
right tool for these small, ill-conditioned least-squares problems — typically ~10
iterations. `"lbfgs"` is a reasonable alternative; `"adam"` is available but needs
thousands of iterations here and is meant for larger, noisy pipelines. Thickness and
oscillator fits have local minima: start from sensible values (presets, known nominal
thickness) or run a few starting points.

See **`example.ipynb`** for a full demo with plots, loss curves, and comparison to
ground truth. The notebook also covers single-oscillator sweeps, the preset gallery,
composition patterns, and `state_dict` persistence.


