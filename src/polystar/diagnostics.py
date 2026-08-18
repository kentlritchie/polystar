"""
Validation of the Lane-Emden solver against known analytic and reference
solutions (SRS section 11, VAL-001..008).

Analytic cases (exact, closed form):
    n=0: theta(xi) = 1 - xi^2/6,      xi_1 = sqrt(6)
    n=1: theta(xi) = sin(xi)/xi,      xi_1 = pi

Reference cases (no closed form; standard tabulated values, e.g.
Horedt, "Polytropes: Applications in Astrophysics and Related Fields"
(2004), Table 2.1; consistent with Chandrasekhar 1939):
    n=1.5: xi_1 = 3.65375,  -xi_1^2 theta'(xi_1) = 2.71406
    n=3.0: xi_1 = 6.89685,  -xi_1^2 theta'(xi_1) = 2.01824
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .lane_emden import solve_lane_emden
from .models import LaneEmdenSolution, SolverConfig

ANALYTIC_XI1: dict[float, float] = {0.0: math.sqrt(6.0), 1.0: math.pi}

REFERENCE_VALUES: dict[float, tuple[float, float]] = {
    1.5: (3.65375, 2.71406),
    3.0: (6.89685, 2.01824),
}


def theta_analytic_n0(xi: np.ndarray) -> np.ndarray:
    return 1.0 - xi**2 / 6.0


def theta_analytic_n1(xi: np.ndarray) -> np.ndarray:
    return np.sin(xi) / xi


@dataclass(frozen=True)
class ValidationResult:
    label: str
    passed: bool
    details: dict[str, float]


def validate_analytic(n: float, config: SolverConfig | None = None, tol: float = 1e-6) -> ValidationResult:
    """VAL-001..004: compare the numerical solution to a closed-form case."""
    if n not in ANALYTIC_XI1:
        raise ValueError(f"No analytic reference for n={n}; use n=0 or n=1")

    solution = solve_lane_emden(n, config)
    if n == 0.0:
        theta_exact = theta_analytic_n0(solution.xi)
    else:
        theta_exact = theta_analytic_n1(solution.xi)

    max_abs_error = float(np.max(np.abs(solution.theta - theta_exact)))
    xi1_expected = ANALYTIC_XI1[n]
    xi1_computed = solution.xi_surface
    xi1_abs_error = abs(xi1_computed - xi1_expected) if xi1_computed is not None else math.inf
    xi1_rel_error = xi1_abs_error / xi1_expected

    passed = solution.solver_success and xi1_rel_error < tol and max_abs_error < 1e-4

    return ValidationResult(
        label=f"n={n} analytic",
        passed=passed,
        details={
            "max_abs_theta_error": max_abs_error,
            "xi1_expected": xi1_expected,
            "xi1_computed": xi1_computed if xi1_computed is not None else float("nan"),
            "xi1_abs_error": xi1_abs_error,
            "xi1_rel_error": xi1_rel_error,
        },
    )


def validate_reference(n: float, config: SolverConfig | None = None, tol: float = 1e-3) -> ValidationResult:
    """VAL-005: compare xi_1 and the surface mass coefficient to trusted
    tabulated values for indices without a closed-form solution."""
    if n not in REFERENCE_VALUES:
        raise ValueError(f"No reference value stored for n={n}; use n=1.5 or n=3.0")

    xi1_expected, mass_coeff_expected = REFERENCE_VALUES[n]
    solution = solve_lane_emden(n, config)

    xi1_computed = solution.xi_surface
    mass_coeff_computed = solution.surface_mass_coefficient

    xi1_rel_error = abs(xi1_computed - xi1_expected) / xi1_expected if xi1_computed else math.inf
    mass_rel_error = (
        abs(mass_coeff_computed - mass_coeff_expected) / mass_coeff_expected
        if mass_coeff_computed is not None
        else math.inf
    )

    passed = solution.solver_success and xi1_rel_error < tol and mass_rel_error < tol

    return ValidationResult(
        label=f"n={n} reference",
        passed=passed,
        details={
            "xi1_expected": xi1_expected,
            "xi1_computed": xi1_computed if xi1_computed is not None else float("nan"),
            "xi1_rel_error": xi1_rel_error,
            "mass_coeff_expected": mass_coeff_expected,
            "mass_coeff_computed": mass_coeff_computed if mass_coeff_computed is not None else float("nan"),
            "mass_coeff_rel_error": mass_rel_error,
        },
    )


def lane_emden_residual(solution: LaneEmdenSolution, edge_fraction: float = 0.02) -> np.ndarray:
    """VAL-006: evaluate (1/xi^2) d/dxi[xi^2 dtheta/dxi] + theta^n away from
    xi=0, using finite differences of the solver's own dense output. Should
    be small everywhere in the interior. `edge_fraction` excludes both ends
    of the domain: near xi=0 the 1/xi^2 factor amplifies ordinary finite-
    difference truncation error (even though the ODE itself is regularized
    there via the series expansion), and near the surface theta -> 0 makes
    the residual's relative scale ill-defined."""
    xi = solution.xi
    dtheta = solution.dtheta_dxi
    flux = xi**2 * dtheta
    d_flux_dxi = np.gradient(flux, xi)
    theta_pos = np.clip(solution.theta, 0.0, None)
    forcing = theta_pos**solution.n if solution.n != 0 else np.ones_like(theta_pos)
    residual = d_flux_dxi / xi**2 + forcing

    n_edge = max(int(edge_fraction * len(xi)), 2)
    return residual[n_edge:-n_edge]


def mass_is_monotonic(solution: LaneEmdenSolution) -> bool:
    """VAL-008: enclosed dimensionless mass must be non-decreasing outward."""
    diffs = np.diff(solution.mass_dimensionless)
    return bool(np.all(diffs >= -1e-9))
