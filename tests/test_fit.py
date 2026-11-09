"""Tests for fit(): synthetic-data recovery, methods, bounds, statistics, API errors."""

import math

import pytest
import torch
import torch.nn as nn

from dispertorch import (Cauchy, ConstantEps, ConstantIndex, Layer, Lorentz, Measurement,
                         Stack, fit)

F64 = torch.float64
WL = torch.linspace(400.0, 1000.0, 61, dtype=F64)
ANGLES_DEG = torch.tensor([65.0, 70.0, 75.0], dtype=F64)
TRUTH = {"d": 250.0, "A": 1.46, "B": 0.0045}


def P(v):
    return nn.Parameter(torch.tensor(v, dtype=F64))


def film_stack(d, A, B, dtype=F64):
    film = Cauchy(P(A) if dtype == F64 else nn.Parameter(torch.tensor(A)),
                  P(B) if dtype == F64 else nn.Parameter(torch.tensor(B)), dtype=dtype)
    d = P(d) if dtype == F64 else nn.Parameter(torch.tensor(d))
    return Stack(ConstantIndex(1.0, dtype=dtype), [Layer(film, d)],
                 ConstantIndex(3.88, 0.02, dtype=dtype))


def synthetic(noise=0.0, seed=0):
    """VASE psi/delta at 65/70/75 deg (degrees) + normal-incidence R."""
    truth = film_stack(**TRUTH)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        res = truth(WL[:, None], torch.deg2rad(ANGLES_DEG)[None, :])
        psi, delta = torch.rad2deg(res.psi), torch.rad2deg(res.delta)
        R = truth(WL, 0.0).R
    psi = psi + noise * torch.randn(psi.shape, generator=g, dtype=F64)
    delta = delta + noise * torch.randn(delta.shape, generator=g, dtype=F64)
    R = R + noise * 1e-2 * torch.randn(R.shape, generator=g, dtype=F64)
    sig = max(noise, 1e-3)
    return [
        Measurement("psi", WL[:, None], psi, ANGLES_DEG[None, :], sigma=sig, degrees=True),
        Measurement("delta", WL[:, None], delta, ANGLES_DEG[None, :], sigma=sig, degrees=True),
        Measurement("R", WL, R, 0.0, sigma=sig * 1e-2),
    ]


def values(stack):
    return {"d": stack.layers[0].d.item(),
            "A": stack.layers[0].model.A.item(),
            "B": stack.layers[0].model.B.item()}


# ── recovery ───────────────────────────────────────────────────────────────

class TestRecovery:
    def test_lm_stack_vase_plus_R(self):
        stack = film_stack(d=220.0, A=1.52, B=0.003)
        res = fit(stack, synthetic())
        assert res.success, res.message
        for k, v in values(stack).items():
            assert v == pytest.approx(TRUTH[k], rel=1e-7), k
        assert res.cost < 1e-12
        assert res.n_iter < 50

    def test_lm_dispersion_model_nk(self):
        truth = ConstantEps(2.0, dtype=F64) + Lorentz(4.0, 4.5, 0.4, dtype=F64)
        with torch.no_grad():
            n, k = truth.nk(WL)
        model = ConstantEps(P(1.7), dtype=F64) + Lorentz(P(3.0), P(4.0), P(0.6), dtype=F64)
        res = fit(model, [Measurement("n", WL, n), Measurement("k", WL, k)])
        assert res.success
        got = [model.terms[0].eps_re, model.terms[1].A, model.terms[1].omega_0,
               model.terms[1].gamma]
        for g_, t in zip(got, [2.0, 4.0, 4.5, 0.4]):
            assert g_.item() == pytest.approx(t, rel=1e-6)

    def test_eps_kinds(self):
        truth = Lorentz(4.0, 4.5, 0.4, dtype=F64)
        with torch.no_grad():
            eps = truth.permittivity(WL)
        model = Lorentz(P(3.0), P(4.2), P(0.5), dtype=F64)
        fit(model, [Measurement("eps1", WL, eps.real), Measurement("eps2", WL, eps.imag)])
        assert model.omega_0.item() == pytest.approx(4.5, rel=1e-8)

    def test_lbfgs(self):
        stack = film_stack(d=240.0, A=1.48, B=0.004)
        res = fit(stack, synthetic(), method="lbfgs", max_iter=500)
        for k, v in values(stack).items():
            assert v == pytest.approx(TRUTH[k], rel=1e-4), k
        assert res.history[-1] < res.history[0]

    def test_adam_decreases_cost(self):
        stack = film_stack(d=240.0, A=1.48, B=0.004)
        res = fit(stack, synthetic(), method="adam", max_iter=200, lr=1e-3)
        assert res.n_iter > 10        # regression: must not stop on the first step
        assert res.history[-1] < 0.5 * res.history[0]

    def test_float32_system_is_updated_in_place(self):
        stack = film_stack(d=230.0, A=1.5, B=0.004, dtype=torch.float32)
        fit(stack, synthetic())
        assert stack.layers[0].d.dtype == torch.float32
        assert stack.layers[0].d.item() == pytest.approx(TRUTH["d"], rel=1e-5)


# ── statistics ─────────────────────────────────────────────────────────────

class TestStatistics:
    def test_noisy_reduced_chi2_and_stderr(self):
        stack = film_stack(d=230.0, A=1.5, B=0.004)
        res = fit(stack, synthetic(noise=0.02, seed=1))
        assert 0.8 < res.reduced_chi2 < 1.2
        v = values(stack)
        names = {"d": "layers.0.d", "A": "layers.0.model.A", "B": "layers.0.model.B"}
        for k, n in names.items():
            err = res.stderr[n].item()
            assert err > 0
            assert abs(v[k] - TRUTH[k]) < 5 * err, k
        C = res.correlation
        assert torch.allclose(torch.diagonal(C), torch.ones(3, dtype=F64))
        assert (C.abs() <= 1 + 1e-9).all()
        assert "layers.0.d" in res.summary()

    def test_residuals_per_measurement(self):
        res = fit(film_stack(d=230.0, A=1.5, B=0.004), synthetic())
        assert set(res.residuals) == {"psi", "delta", "R"}
        assert res.residuals["psi"].numel() == WL.numel() * ANGLES_DEG.numel()
        assert res.n_data == 2 * 61 * 3 + 61


# ── options ────────────────────────────────────────────────────────────────

class TestOptions:
    def test_bounds_by_object_and_name(self):
        for key in ("layers.0.d", None):
            stack = film_stack(d=220.0, A=1.5, B=0.004)
            k = key if key else stack.layers[0].d
            fit(stack, synthetic(), bounds={k: (None, 240.0)})
            assert stack.layers[0].d.item() == pytest.approx(240.0, abs=1e-9)

    def test_params_subset(self):
        stack = film_stack(d=230.0, A=TRUTH["A"], B=0.004)
        res = fit(stack, synthetic(), params=["layers.0.d"])
        assert stack.layers[0].model.B.item() == 0.004
        assert stack.layers[0].model.A.item() == TRUTH["A"]
        assert list(res.params) == ["layers.0.d"]

    def test_delta_wrapping(self):
        meas = synthetic()
        meas[1] = Measurement("delta", meas[1].wl, meas[1].data + 360.0, meas[1].theta,
                              sigma=meas[1].sigma, degrees=True)
        stack = film_stack(d=230.0, A=1.5, B=0.004)
        res = fit(stack, meas)
        assert res.cost < 1e-12
        assert stack.layers[0].d.item() == pytest.approx(TRUTH["d"], rel=1e-7)

    def test_radians_equivalent_to_degrees(self):
        truth = film_stack(**TRUTH)
        th = math.radians(70)
        with torch.no_grad():
            psi = truth(WL, th).psi
        stack = film_stack(d=240.0, A=1.5, B=0.004)
        fit(stack, Measurement("psi", WL, psi, th))
        assert stack.layers[0].d.item() == pytest.approx(TRUTH["d"], rel=1e-6)

    def test_callable_kind(self):
        truth = film_stack(**TRUTH)
        with torch.no_grad():
            tan_psi = torch.tan(truth(WL, 1.2).psi)

        def tanpsi(system, wl, theta):
            return torch.tan(system(wl, theta).psi)

        stack = film_stack(d=240.0, A=1.5, B=0.004)
        fit(stack, Measurement(tanpsi, WL, tan_psi, 1.2))
        assert stack.layers[0].d.item() == pytest.approx(TRUTH["d"], rel=1e-6)

    def test_max_iter_respected(self):
        res = fit(film_stack(d=200.0, A=1.6, B=0.001), synthetic(), max_iter=2)
        assert res.n_iter <= 2 and not res.success


# ── errors ─────────────────────────────────────────────────────────────────

class TestErrors:
    def test_unknown_kind(self):
        with pytest.raises(ValueError, match="unknown kind"):
            Measurement("Q", WL, WL)

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="unknown method"):
            fit(film_stack(**TRUTH), synthetic(), method="newton")

    def test_kind_system_mismatch(self):
        with pytest.raises(ValueError, match="DispersionModel"):
            fit(film_stack(**TRUTH), Measurement("n", WL, WL))
        with pytest.raises(ValueError, match="Stack"):
            fit(Cauchy(P(1.5), P(0.004), dtype=F64), Measurement("R", WL, WL))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            fit(film_stack(**TRUTH), Measurement("R", WL, WL[:10]))

    def test_no_free_parameters(self):
        frozen = Stack(ConstantIndex(1.0), [(ConstantIndex(1.5), 100.0)], ConstantIndex(3.9))
        with pytest.raises(ValueError, match="no free parameters"):
            fit(frozen, Measurement("R", WL, WL))

    def test_bad_param_name(self):
        with pytest.raises(ValueError, match="no parameter named"):
            fit(film_stack(**TRUTH), synthetic(), params=["layers.0.dd"])


# ── devices and dtypes ─────────────────────────────────────────────────────

ACCELERATORS = [d for d, ok in (("cuda", torch.cuda.is_available()),
                                ("mps", torch.backends.mps.is_available())) if ok]


def _f32_problem():
    """float32 system (moveable to MPS) and CPU float64 data."""
    truth = film_stack(**TRUTH)
    with torch.no_grad():
        R = truth(WL, 0.0).R
        psi = truth(WL, 1.2).psi
    stack = film_stack(d=240.0, A=1.5, B=0.004, dtype=torch.float32)
    meas = [Measurement("R", WL, R, sigma=1e-3), Measurement("psi", WL, psi, 1.2, sigma=1e-3)]
    return stack, meas


class TestDevices:
    def test_float32_working_precision_on_cpu(self):
        stack, meas = _f32_problem()
        res = fit(stack, meas, dtype=torch.float32)
        assert stack.layers[0].d.item() == pytest.approx(TRUTH["d"], rel=1e-4)
        assert res.params["layers.0.d"].device.type == "cpu"

    @pytest.mark.parametrize("device", ACCELERATORS)
    @pytest.mark.parametrize("method", ["lm", "lbfgs"])
    def test_fit_on_accelerator_with_cpu_data(self, device, method):
        stack, meas = _f32_problem()
        stack.to(device)
        bound = torch.tensor(260.0)                       # CPU bound tensor, on purpose
        res = fit(stack, meas, method=method, max_iter=300,
                  bounds={"layers.0.d": (0.0, bound)})
        d = stack.layers[0].d
        assert d.device.type == device and d.dtype == torch.float32
        assert d.item() == pytest.approx(TRUTH["d"], rel=1e-3)
        assert all(t.device.type == "cpu" for t in res.stderr.values())
        assert torch.isfinite(res.correlation).all()

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
    def test_float64_on_mps_is_rejected_clearly(self):
        stack, meas = _f32_problem()
        stack.to("mps")
        with pytest.raises(ValueError, match="MPS does not support float64"):
            fit(stack, meas, dtype=F64)

    def test_parameters_on_mixed_devices_rejected(self):
        if not ACCELERATORS:
            pytest.skip("needs an accelerator")
        stack, meas = _f32_problem()
        stack.layers[0].model.to(ACCELERATORS[0])        # film on GPU, thickness on CPU
        with pytest.raises(ValueError, match="several devices"):
            fit(stack, meas)
