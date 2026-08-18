"""Structured data objects for solved models (SRS section 8, Appendix section)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SolverConfig:
    """Numerical parameters for the Lane-Emden integration (NR-001, AR-007).

    Attributes
    ----------
    xi_0 : float
        Small positive starting coordinate used to sidestep the xi=0
        singularity; the series expansion supplies the initial condition
        there instead of evaluating the ODE's 1/xi term directly.
    xi_max : float
        Upper integration bound. Increase for n approaching 5, where the
        surface (if any) can lie far out or not exist at all.
    rtol, atol : float
        Relative/absolute tolerances passed to scipy.integrate.solve_ivp.
    method : str
        solve_ivp integration method.
    n_dense : int
        Number of points in the returned dense output grid (NR-005).
    """

    xi_0: float = 1e-6
    xi_max: float = 50.0
    rtol: float = 1e-10
    atol: float = 1e-12
    method: str = "RK45"
    n_dense: int = 2000


@dataclass(frozen=True)
class LaneEmdenSolution:
    """Result of solving the Lane-Emden equation for one polytropic index n.

    All arrays are dimensionless and sampled on a common dense xi grid
    from xi_0 up to xi_surface (or xi_max if no finite surface was found).
    """

    n: float
    xi: np.ndarray
    theta: np.ndarray
    dtheta_dxi: np.ndarray
    xi_surface: float | None
    rho_normalized: np.ndarray
    mass_dimensionless: np.ndarray
    surface_mass_coefficient: float | None
    solver_success: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def central_concentration(self) -> float | None:
        """rho_c / <rho> = xi_1^3 / (3 * (-xi_1^2 theta'(xi_1))) (Appendix A.2)."""
        if self.xi_surface is None or self.surface_mass_coefficient in (None, 0.0):
            return None
        return self.xi_surface**3 / (3.0 * self.surface_mass_coefficient)


@dataclass(frozen=True)
class PhysicalProfile:
    """A LaneEmdenSolution scaled to physical (SI) units via rho_c and K."""

    n: float
    rho_c: float
    K: float
    scale_length: float
    r: np.ndarray
    rho: np.ndarray
    pressure: np.ndarray
    enclosed_mass: np.ndarray
    radius: float
    total_mass: float
    units: str = "SI (m, kg, Pa)"
