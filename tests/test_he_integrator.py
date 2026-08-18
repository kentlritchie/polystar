"""HE integrator vs. real Model S data (pulled forward from SRS 18.1)."""

import numpy as np
import pytest

from polystar.he_integrator import compare_to_model_s, integrate_hydrostatic_equilibrium
from polystar.model_s_data import load_model_s


@pytest.fixture(scope="module")
def model():
    return load_model_s()


def test_model_s_mu_mode_reproduces_model_s_closely(model):
    # mu(r) here is derived from Model S's own P, rho, T, so this is a
    # correctness check on the integrator, not an independent prediction.
    result = integrate_hydrostatic_equilibrium(model, mu_mode="model_s")
    assert result.solver_success
    errors = compare_to_model_s(result, model)
    interior = slice(5, -5)  # skip the least-constrained near-center/edge points
    assert np.nanmax(np.abs(errors["rho_relative_error"][interior])) < 0.05
    assert np.nanmax(np.abs(errors["pressure_relative_error"][interior])) < 0.05


def test_constant_mu_mode_runs_and_stays_finite(model):
    result = integrate_hydrostatic_equilibrium(model, mu_mode="constant", mu_constant=0.62)
    assert result.solver_success
    assert np.all(np.isfinite(result.pressure))
    assert np.all(np.isfinite(result.density))
    assert np.all(result.pressure >= 0)


def test_constant_mu_deviates_more_than_model_s_mu_in_envelope(model):
    result_const = integrate_hydrostatic_equilibrium(model, mu_mode="constant")
    result_real = integrate_hydrostatic_equilibrium(model, mu_mode="model_s")
    err_const = compare_to_model_s(result_const, model)
    err_real = compare_to_model_s(result_real, model)
    outer = slice(int(0.8 * len(result_const.r)), None)
    const_outer_error = np.nanmean(np.abs(err_const["rho_relative_error"][outer]))
    real_outer_error = np.nanmean(np.abs(err_real["rho_relative_error"][outer]))
    assert const_outer_error > real_outer_error


def test_enclosed_mass_monotonic(model):
    result = integrate_hydrostatic_equilibrium(model, mu_mode="model_s")
    assert np.all(np.diff(result.enclosed_mass) >= -1e-6)
