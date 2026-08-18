"""
Non-polytropic hydrostatic-equilibrium integrator (SRS section 18.1, pulled
forward from "stretch goals" into v1 alongside the Lane-Emden solver).

Unlike lane_emden.py, this does NOT assume a polytropic P-rho power law.
Instead it integrates the two fundamental structure equations directly,

    dP/dr = -G m(r) rho(r) / r^2      (hydrostatic equilibrium)
    dm/dr = 4 pi r^2 rho(r)            (mass continuity)

closed with the ideal-gas law rho = P mu m_H / (k_B T), using the REAL
temperature profile T(r) from Model S (model_s_data.py) as an external
input rather than solving radiative transport. This is the honest scope
limit of this module: energy transport/generation are not modeled, so the
result is only as good as (a) the ideal-gas approximation and (b) whichever
mean molecular weight mu(r) is supplied.

Two mu modes are offered so the comparison against real Model S is
informative rather than circular:

- "constant": a single typical solar-interior mu (~0.6 for fully ionized
  H/He), a genuinely independent assumption. Deviates from Model S in the
  outer envelope, where partial ionization raises mu.
- "model_s": mu(r) implied by Model S's own P, rho, T via the ideal-gas
  law. Nearly perfectly reproduces Model S's P(r)/rho(r) by construction —
  useful as a sanity check that the integrator itself is correct, not as
  an independent prediction.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp

from .constants import K_B, M_H, G
from .model_s_data import ModelSProfile

MuMode = Literal["constant", "model_s"]

DEFAULT_MU_CONSTANT = 0.62  # typical fully-ionized solar-interior mean molecular weight


@dataclass(frozen=True)
class HEIntegrationResult:
    """Output of integrate_hydrostatic_equilibrium."""

    r: np.ndarray  # m
    pressure: np.ndarray  # Pa
    density: np.ndarray  # kg/m^3
    enclosed_mass: np.ndarray  # kg
    mu_mode: MuMode
    mu_constant: float | None
    r_surface: float | None  # m, where P first reaches ~0 (None if it never does)
    solver_success: bool
    solver_message: str


def _make_rhs(
    T_of_r: Callable[[float], float],
    mu_of_r: Callable[[float], float] | None,
    mu_constant: float | None,
) -> Callable[[float, np.ndarray], list[float]]:
    def rhs(r: float, y: np.ndarray) -> list[float]:
        P, m = y
        P_pos = max(P, 0.0)
        T = float(T_of_r(r))
        mu = float(mu_of_r(r)) if mu_of_r is not None else mu_constant
        rho = P_pos * mu * M_H / (K_B * T)
        dP_dr = -G * m * rho / r**2
        dm_dr = 4.0 * math.pi * r**2 * rho
        return [dP_dr, dm_dr]

    return rhs


def _surface_event(r: float, y: np.ndarray) -> float:
    return y[0]


_surface_event.terminal = True
_surface_event.direction = -1.0


def integrate_hydrostatic_equilibrium(
    model: ModelSProfile,
    mu_mode: MuMode = "constant",
    mu_constant: float = DEFAULT_MU_CONSTANT,
    r_start_fraction: float = 1e-4,
    n_dense: int = 2000,
    rtol: float = 1e-8,
    atol: float = 1e-6,
) -> HEIntegrationResult:
    """Integrate HE + mass continuity outward from the center, closed with
    the ideal-gas law and Model S's real T(r), starting from Model S's real
    central pressure and density (so the two models share a starting point
    and any divergence downstream is due to the mu(r)/EOS assumption, not a
    mismatched boundary condition).
    """
    T_of_r = model.temperature_of_r()
    mu_of_r = model.mu_of_r() if mu_mode == "model_s" else None
    mu_const = None if mu_mode == "model_s" else mu_constant

    rho_c = float(model.rho[0])
    P_c = float(model.pressure[0])
    r_eps = r_start_fraction * model.radius
    m_eps = (4.0 / 3.0) * math.pi * r_eps**3 * rho_c

    rhs = _make_rhs(T_of_r, mu_of_r, mu_const)

    result = solve_ivp(
        rhs,
        t_span=(r_eps, model.radius),
        y0=[P_c, m_eps],
        method="RK45",
        rtol=rtol,
        atol=atol,
        events=_surface_event,
        dense_output=True,
    )

    surface_found = len(result.t_events[0]) > 0
    r_surface = float(result.t_events[0][0]) if surface_found else None
    r_end = r_surface if surface_found else model.radius

    r_dense = np.linspace(r_eps, r_end, n_dense)
    P_dense, m_dense = result.sol(r_dense)
    P_dense = np.clip(P_dense, 0.0, None)

    T_dense = T_of_r(r_dense)
    mu_dense = mu_of_r(r_dense) if mu_of_r is not None else mu_const
    rho_dense = P_dense * mu_dense * M_H / (K_B * T_dense)

    return HEIntegrationResult(
        r=r_dense,
        pressure=P_dense,
        density=rho_dense,
        enclosed_mass=m_dense,
        mu_mode=mu_mode,
        mu_constant=mu_constant if mu_mode == "constant" else None,
        r_surface=r_surface,
        solver_success=result.success,
        solver_message=result.message,
    )


def compare_to_model_s(result: HEIntegrationResult, model: ModelSProfile) -> dict[str, np.ndarray]:
    """Relative deviation of the integrated profile from real Model S,
    evaluated on the integrator's own radial grid (VR/VAL-style diagnostic,
    analogous to the polytropic residual check but against real data)."""
    rho_model = model.rho_of_r()(result.r)
    P_model = model.pressure_of_r()(result.r)
    m_model = model.mass_of_r()(result.r)

    with np.errstate(divide="ignore", invalid="ignore"):
        rho_rel_error = np.where(rho_model > 0, (result.density - rho_model) / rho_model, np.nan)
        P_rel_error = np.where(P_model > 0, (result.pressure - P_model) / P_model, np.nan)
        m_rel_error = np.where(m_model > 0, (result.enclosed_mass - m_model) / m_model, np.nan)

    return {
        "rho_relative_error": rho_rel_error,
        "pressure_relative_error": P_rel_error,
        "mass_relative_error": m_rel_error,
    }
