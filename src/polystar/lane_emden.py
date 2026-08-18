"""
Core Lane-Emden solver.

Physics
-------
A polytropic star (P = K rho^(1 + 1/n)) in Newtonian hydrostatic equilibrium
reduces, after the standard dimensionless substitution rho = rho_c theta^n
and r = a*xi, to:

    (1/xi^2) d/dxi [ xi^2 dtheta/dxi ] = -theta^n
    theta(0) = 1,   dtheta/dxi(0) = 0

theta is the dimensionless density-potential function; the first positive
zero xi_1 marks the stellar surface (finite for n < 5).

Numerical treatment
--------------------
The equation is singular at xi=0 (division by xi^2), so integration starts
at a small xi_0 > 0 using the central series expansion

    theta(xi)       ~= 1 - xi^2/6 + n*xi^4/120
    dtheta/dxi(xi)  ~= -xi/3 + n*xi^3/30

rather than evaluating the ODE right-hand side at xi=0. Reference:
Chandrasekhar (1939), "An Introduction to the Study of Stellar Structure".
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .models import LaneEmdenSolution, SolverConfig


class LaneEmdenError(ValueError):
    """Raised for unsupported inputs or solver failures (FR-001, NR-008)."""


def _central_series(xi0: float, n: float) -> tuple[float, float]:
    """theta and dtheta/dxi at xi0, from the central series expansion."""
    theta0 = 1.0 - xi0**2 / 6.0 + n * xi0**4 / 120.0
    dtheta0 = -xi0 / 3.0 + n * xi0**3 / 30.0
    return theta0, dtheta0


def _rhs(xi: float, y: np.ndarray, n: float) -> list[float]:
    """First-order system: y = [theta, dtheta/dxi] (FR-002).

    theta is clamped to >= 0 before raising to power n so that fractional
    n never hits a negative base (NR-004); this only matters in the last
    fraction of a step before the surface-crossing event terminates the
    integration.
    """
    theta, dtheta = y
    theta_pos = max(0.0, theta)
    forcing = theta_pos**n if n != 0 else 1.0
    d2theta = -forcing - 2.0 * dtheta / xi
    return [dtheta, d2theta]


def _surface_event(xi: float, y: np.ndarray, n: float) -> float:
    return y[0]


_surface_event.terminal = True
_surface_event.direction = -1.0


def solve_lane_emden(n: float, config: SolverConfig | None = None) -> LaneEmdenSolution:
    """Solve the Lane-Emden equation for polytropic index n.

    Parameters
    ----------
    n : float
        Polytropic index. Must be >= 0 (FR-001, TEST-006); this version
        does not support negative indices.
    config : SolverConfig, optional
        Numerical parameters (start radius, tolerances, integration
        method, output density). Defaults are chosen to recover the
        n=0/n=1 analytic surfaces to better than 1e-6 relative error
        (NR-002).

    Returns
    -------
    LaneEmdenSolution
        Dense dimensionless profile plus the detected surface (if any).

    Raises
    ------
    LaneEmdenError
        If n is invalid, or the ODE integration itself fails (NR-008).
    """
    if not np.isfinite(n) or n < 0:
        raise LaneEmdenError(f"n must be a finite value >= 0, got {n!r}")

    cfg = config or SolverConfig()
    theta0, dtheta0 = _central_series(cfg.xi_0, n)

    result = solve_ivp(
        _rhs,
        t_span=(cfg.xi_0, cfg.xi_max),
        y0=[theta0, dtheta0],
        args=(n,),
        method=cfg.method,
        rtol=cfg.rtol,
        atol=cfg.atol,
        events=_surface_event,
        dense_output=True,
        t_eval=None,
    )

    if not result.success:
        raise LaneEmdenError(f"Lane-Emden integration failed for n={n}: {result.message}")

    surface_found = len(result.t_events[0]) > 0
    xi_surface = float(result.t_events[0][0]) if surface_found else None

    # Dense, uniform output grid over the physical domain (NR-005), regardless
    # of where the adaptive solver actually placed its internal steps.
    xi_end = xi_surface if surface_found else cfg.xi_max
    xi_dense = np.linspace(cfg.xi_0, xi_end, cfg.n_dense)
    theta_dense, dtheta_dense = result.sol(xi_dense)

    # Guard against tiny negative overshoot right at the surface from
    # interpolation of the dense solution (NR-004).
    theta_dense = np.where(np.abs(theta_dense) < 1e-13, 0.0, theta_dense)
    theta_clamped = np.clip(theta_dense, 0.0, None)

    rho_normalized = theta_clamped**n if n != 0 else np.ones_like(theta_clamped)
    mass_dimensionless = -(xi_dense**2) * dtheta_dense

    surface_mass_coefficient = None
    if surface_found:
        _theta_at_surface, dtheta_at_surface = result.sol(xi_surface)
        surface_mass_coefficient = float(-(xi_surface**2) * dtheta_at_surface)

    metadata = {
        "surface_found": surface_found,
        "method": cfg.method,
        "rtol": cfg.rtol,
        "atol": cfg.atol,
        "xi_0": cfg.xi_0,
        "xi_max": cfg.xi_max,
        "solver_message": result.message,
        "n_steps": int(result.t.size),
    }
    if n >= 4.5 and not surface_found:
        metadata["warning"] = (
            f"No surface zero found for n={n} within xi_max={cfg.xi_max}. "
            "For n approaching 5 the radius grows rapidly (n=5 is formally "
            "infinite); increase SolverConfig.xi_max if a finite surface is "
            "expected, or treat this model as surface-less."
        )

    return LaneEmdenSolution(
        n=n,
        xi=xi_dense,
        theta=theta_clamped,
        dtheta_dxi=dtheta_dense,
        xi_surface=xi_surface,
        rho_normalized=rho_normalized,
        mass_dimensionless=mass_dimensionless,
        surface_mass_coefficient=surface_mass_coefficient,
        solver_success=result.success,
        metadata=metadata,
    )
