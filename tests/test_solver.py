"""TEST-001..003, 005..008: solver internals and edge-case behavior."""

import math

import numpy as np
import pytest

from polystar.diagnostics import lane_emden_residual, mass_is_monotonic
from polystar.lane_emden import LaneEmdenError, _central_series, _rhs, solve_lane_emden
from polystar.models import SolverConfig


def test_central_series_matches_direct_formula():
    xi0, n = 1e-3, 1.5
    theta0, dtheta0 = _central_series(xi0, n)
    assert math.isclose(theta0, 1.0 - xi0**2 / 6.0 + n * xi0**4 / 120.0)
    assert math.isclose(dtheta0, -xi0 / 3.0 + n * xi0**3 / 30.0)


def test_rhs_known_point():
    # At theta=1, dtheta=0, xi=1, n=0: d2theta = -1 - 0 = -1
    dtheta, d2theta = _rhs(1.0, [1.0, 0.0], 0.0)
    assert dtheta == 0.0
    assert math.isclose(d2theta, -1.0)


def test_rhs_clamps_negative_theta():
    # Non-integer n with negative theta must not raise/produce NaN.
    _dtheta, d2theta = _rhs(1.0, [-0.1, 0.0], 1.5)
    assert np.isfinite(d2theta)


def test_density_and_mass_derived_quantities():
    solution = solve_lane_emden(1.5)
    assert np.all(solution.rho_normalized >= 0.0)
    assert np.all(solution.rho_normalized <= 1.0 + 1e-9)
    assert mass_is_monotonic(solution)


def test_n0_analytic_surface():
    solution = solve_lane_emden(0.0)
    assert solution.xi_surface is not None
    assert math.isclose(solution.xi_surface, math.sqrt(6.0), rel_tol=1e-6)


def test_n1_analytic_surface():
    solution = solve_lane_emden(1.0)
    assert solution.xi_surface is not None
    assert math.isclose(solution.xi_surface, math.pi, rel_tol=1e-6)


def test_surface_event_stops_at_first_zero():
    solution = solve_lane_emden(3.0)
    assert solution.xi_surface is not None
    # theta should not go meaningfully negative past the detected surface
    assert solution.theta[-1] >= -1e-6


def test_negative_n_rejected():
    with pytest.raises(LaneEmdenError):
        solve_lane_emden(-1.0)


def test_n_near_five_no_nans():
    solution = solve_lane_emden(4.9, config=SolverConfig(xi_max=60.0))
    assert solution.solver_success
    assert np.all(np.isfinite(solution.theta))
    assert np.all(np.isfinite(solution.rho_normalized))


def test_n_five_reports_no_surface_within_default_range():
    solution = solve_lane_emden(5.0, config=SolverConfig(xi_max=20.0))
    # n=5 is formally infinite-radius; within a modest xi_max we should not
    # falsely detect a finite surface, and metadata should say so.
    assert solution.xi_surface is None
    assert "warning" in solution.metadata


def test_lane_emden_residual_is_small():
    solution = solve_lane_emden(1.5)
    residual = lane_emden_residual(solution)
    assert np.all(np.isfinite(residual))
    assert np.max(np.abs(residual)) < 1e-3
