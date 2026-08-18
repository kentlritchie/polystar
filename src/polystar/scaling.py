"""
Physical scaling: map a dimensionless LaneEmdenSolution to SI-unit profiles
given a central density rho_c and polytropic constant K (SRS section 6.3).

    a^2 = (n + 1) K rho_c^(1/n - 1) / (4 pi G)
    r = a xi,  R = a xi_1
    rho(r) = rho_c theta^n
    P(r)   = K rho(r)^(1 + 1/n)
    M(r)   = 4 pi a^3 rho_c [-xi^2 dtheta/dxi]

n = 0 is excluded: the polytropic relation P = K rho^(1 + 1/n) and the scale
length above both require 1/n, which is undefined at n=0. A theta=0 model
is physically a uniform-density sphere, not a genuine power-law polytrope,
so it has no well-defined K; only its dimensionless structure is available.
"""

from __future__ import annotations

import math

from .constants import G
from .models import LaneEmdenSolution, PhysicalProfile


class ScalingError(ValueError):
    """Raised when physical scaling is requested for an unsupported model."""


def scale_to_physical(
    solution: LaneEmdenSolution, rho_c: float, K: float
) -> PhysicalProfile:
    """Convert a dimensionless Lane-Emden solution to SI-unit profiles.

    Parameters
    ----------
    solution : LaneEmdenSolution
        A solved, finite-radius model (solution.xi_surface must not be None).
    rho_c : float
        Central density [kg/m^3]. Must be positive.
    K : float
        Polytropic constant [SI units consistent with P = K*rho^(1+1/n)].
        Must be positive.
    """
    if solution.n == 0:
        raise ScalingError(
            "Physical scaling is undefined at n=0: P = K*rho^(1+1/n) and the "
            "Lane-Emden scale length both require 1/n. Use the dimensionless "
            "theta/density/mass profile directly for n=0."
        )
    if solution.xi_surface is None:
        raise ScalingError(
            "Cannot scale a model with no detected surface (xi_surface is "
            "None) — increase SolverConfig.xi_max or choose n < 5."
        )
    if rho_c <= 0 or K <= 0:
        raise ScalingError(f"rho_c and K must be positive, got rho_c={rho_c!r}, K={K!r}")

    n = solution.n
    a = math.sqrt((n + 1.0) * K * rho_c ** (1.0 / n - 1.0) / (4.0 * math.pi * G))

    r = a * solution.xi
    rho = rho_c * solution.rho_normalized
    pressure = K * rho ** (1.0 + 1.0 / n)
    enclosed_mass = 4.0 * math.pi * a**3 * rho_c * solution.mass_dimensionless

    radius = a * solution.xi_surface
    total_mass = float(4.0 * math.pi * a**3 * rho_c * solution.surface_mass_coefficient)

    return PhysicalProfile(
        n=n,
        rho_c=rho_c,
        K=K,
        scale_length=a,
        r=r,
        rho=rho,
        pressure=pressure,
        enclosed_mass=enclosed_mass,
        radius=radius,
        total_mass=total_mass,
    )


def K_rho_c_from_mass_radius(n: float, xi_surface: float, mass_coeff: float, M: float, R: float) -> tuple[float, float]:
    """Solve the inverse problem: given a target (M, R) and a solved
    dimensionless model (xi_surface, mass_coeff for that n), recover the
    (rho_c, K) pair that reproduces it. Convenience for 'build a star with
    the Sun's mass and radius' workflows.
    """
    if n == 0:
        raise ScalingError("K is undefined at n=0; supply rho_c directly instead.")
    a = R / xi_surface
    rho_c = M / (4.0 * math.pi * a**3 * mass_coeff)
    K = 4.0 * math.pi * G * a**2 * rho_c ** ((n - 1.0) / n) / (n + 1.0)
    return rho_c, K
