# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`dispertorch` — differentiable refractive-index dispersion models built on `torch.nn.Module`. Permittivity is the primitive; refractive index is derived. The library is designed for gradient-based fitting of optical material parameters.

## Commands

```bash
# Install (with dev dependencies)
uv sync --group dev

# Run tests
uv run pytest

# Run a single test
uv run pytest tests/test_foo.py::test_bar

# Run the scratch entry point
uv run python main.py
```

## Architecture

**Primitive: permittivity, not index.** Every model implements `permittivity(wl: Tensor) -> Tensor` (complex relative permittivity vs. wavelength). `n`, `k`, `nk`, and `index` are all derived on `DispersionModel` and must not be overridden.

**Sign convention:** `eps = eps' + i·eps''` with `eps'' >= 0` for absorbing media (`exp(-i·omega·t)` time convention). The principal square root then gives `k >= 0`.

**Composition via `+`:** `model_a + model_b` returns a `Composite` whose permittivity is their sum. Nested composites are flattened. `sum([...])` works via `__radd__`.

**Coefficient registration (`utils.register`):** Pass an `nn.Parameter` to make a coefficient optimizable; pass a plain scalar/tensor to store it as a non-gradient buffer (still moves with `.to()`, saved in `state_dict`). All coefficients are stored in the declared `dtype` (default `float32`).

**Model registry (`utils.register_model`):** The `@register_model("Name")` decorator registers subclasses in `MODEL_REGISTRY` for lookup by name.

### Key files

| File | Purpose |
|------|---------|
| `src/dispertorch/base.py` | `DispersionModel` base class + `Composite` |
| `src/dispertorch/utils.py` | `register`, `register_model`, `MODEL_REGISTRY` |
| `src/dispertorch/constant.py` | `ConstantIndex`, `ConstantEps` (non-dispersive terms) |
| `src/dispertorch/oscillator.py` | `Drude`, `Lorentz`, `TaucLorentz` (eV-parameterised oscillators) |
| `src/dispertorch/transparent.py` | `Cauchy`, `Sellmeier` (real-index, µm coefficients) |

### Adding a new model

1. Subclass `DispersionModel`.
2. Decorate with `@register_model("ModelName")`.
3. In `__init__`, call `register(self, "param_name", value, dtype)` for each coefficient.
   Names must not shadow the base class's reserved accessors (`n`, `k`, `nk`, `index`,
   `permittivity`, `forward`) — `register` raises `ValueError` if they do.
4. Implement `permittivity(self, wl: Tensor) -> Tensor`.
5. Export from `__init__.py`.
