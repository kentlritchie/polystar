"""Physical-scaling round trip and edge cases (FR-009, FR-010)."""

import math

import pytest

from polystar.lane_emden import solve_lane_emden
from polystar.scaling import K_rho_c_from_mass_radius, ScalingError, scale_to_physical


def test_scale_to_physical_round_trip():
    n = 1.5
    M_target, R_target = 2.0e30, 5.0e8  # arbitrary, not physically the Sun
    solution = solve_lane_emden(n)

    rho_c, K = K_rho_c_from_mass_radius(
        n, solution.xi_surface, solution.surface_mass_coefficient, M_target, R_target
    )
    profile = scale_to_physical(solution, rho_c, K)

    assert math.isclose(profile.radius, R_target, rel_tol=1e-9)
    assert math.isclose(profile.total_mass, M_target, rel_tol=1e-6)
    assert profile.rho[0] == pytest.approx(rho_c)
    assert profile.r[0] >= 0.0


def test_scale_to_physical_pressure_positive_and_decreasing():
    n = 3.0
    solution = solve_lane_emden(n)
    profile = scale_to_physical(solution, rho_c=1.5e5, K=3.8e9)
    assert (profile.pressure >= 0).all()
    # Pressure should be (weakly) monotonically decreasing outward.
    assert (profile.pressure[1:] <= profile.pressure[:-1] + 1e-3).all()


def test_scaling_undefined_at_n0():
    solution = solve_lane_emden(0.0)
    with pytest.raises(ScalingError):
        scale_to_physical(solution, rho_c=1.0, K=1.0)


def test_scaling_requires_finite_surface():
    solution = solve_lane_emden(5.0)  # no surface within default xi_max
    with pytest.raises(ScalingError):
        scale_to_physical(solution, rho_c=1.0, K=1.0)


def test_scaling_rejects_nonpositive_inputs():
    solution = solve_lane_emden(1.5)
    with pytest.raises(ScalingError):
        scale_to_physical(solution, rho_c=-1.0, K=1.0)
    with pytest.raises(ScalingError):
        scale_to_physical(solution, rho_c=1.0, K=0.0)
