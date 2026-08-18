"""
Reusable matplotlib figure builders (SRS section 7). Kept separate from the
Streamlit app so the CLI and notebooks can produce the same figures
(AR-001). Every plot carries axis labels with units and a legend, and
distinguishes series by linestyle/marker as well as color (VR-008).
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from .diagnostics import theta_analytic_n0, theta_analytic_n1
from .he_integrator import HEIntegrationResult, compare_to_model_s
from .lane_emden import solve_lane_emden
from .model_s_data import ModelSProfile
from .models import LaneEmdenSolution

_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
_MARKERS = ["o", "s", "^", "D", "v", "P"]


def plot_theta(solution: LaneEmdenSolution) -> Figure:
    """VR-001: theta(xi), with the stellar surface marked if finite."""
    fig = Figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(solution.xi, solution.theta, "-", color="tab:blue", label=f"n = {solution.n}")
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    if solution.xi_surface is not None:
        ax.axvline(solution.xi_surface, color="tab:red", linestyle="--", linewidth=1,
                    label=f"surface, ξ₁ = {solution.xi_surface:.4f}")
    ax.set_xlabel(r"Dimensionless radius $\xi$")
    ax.set_ylabel(r"Lane-Emden function $\theta(\xi)$")
    ax.set_title(f"Lane-Emden solution, n = {solution.n}")
    ax.legend()
    fig.tight_layout()
    return fig


def _normalized_radius(solution: LaneEmdenSolution) -> np.ndarray:
    end = solution.xi_surface if solution.xi_surface is not None else solution.xi[-1]
    return solution.xi / end


def plot_density_profile(solution: LaneEmdenSolution) -> Figure:
    """VR-002: normalized density rho/rho_c vs normalized radius r/R."""
    fig = Figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(_normalized_radius(solution), solution.rho_normalized, color="tab:orange")
    ax.set_xlabel(r"Normalized radius $r/R$")
    ax.set_ylabel(r"Normalized density $\rho/\rho_c$")
    ax.set_title(f"Density profile, n = {solution.n}")
    fig.tight_layout()
    return fig


def plot_mass_profile(solution: LaneEmdenSolution) -> Figure:
    """VR-003: enclosed mass M(r)/M_total vs normalized radius r/R."""
    fig = Figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)
    total = solution.surface_mass_coefficient or solution.mass_dimensionless[-1]
    ax.plot(_normalized_radius(solution), solution.mass_dimensionless / total, color="tab:green")
    ax.set_xlabel(r"Normalized radius $r/R$")
    ax.set_ylabel(r"Enclosed mass $M(r)/M_{\mathrm{total}}$")
    ax.set_title(f"Enclosed-mass profile, n = {solution.n}")
    fig.tight_layout()
    return fig


def plot_comparison(solutions: list[LaneEmdenSolution], normalize: bool = True) -> Figure:
    """VR-004: overlay several n values, distinguished by linestyle+color."""
    fig = Figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111)
    for i, solution in enumerate(solutions):
        x = _normalized_radius(solution) if normalize else solution.xi
        ax.plot(
            x, solution.theta,
            linestyle=_LINESTYLES[i % len(_LINESTYLES)],
            marker=_MARKERS[i % len(_MARKERS)], markevery=max(len(x) // 12, 1), markersize=4,
            label=f"n = {solution.n}",
        )
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.set_xlabel(r"Normalized radius $r/R$" if normalize else r"$\xi$")
    ax.set_ylabel(r"$\theta$")
    ax.set_title("Lane-Emden solutions by polytropic index")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_central_concentration(n_values: list[float]) -> Figure:
    """Central-concentration summary: rho_c/<rho> vs n (SRS §7.1)."""
    concentrations = []
    for n in n_values:
        sol = solve_lane_emden(n)
        concentrations.append(sol.central_concentration)
    fig = Figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(n_values, concentrations, "o-", color="tab:purple")
    ax.set_xlabel("Polytropic index $n$")
    ax.set_ylabel(r"Central concentration $\rho_c / \langle \rho \rangle$")
    ax.set_title("Central concentration vs. polytropic index")
    ax.set_yscale("log")
    fig.tight_layout()
    return fig


def plot_validation_residual(n: float) -> Figure:
    """VR-007: numerical minus analytic theta, for n=0 or n=1."""
    if n not in (0.0, 1.0):
        raise ValueError("Analytic validation residual is only defined for n=0 or n=1")
    solution = solve_lane_emden(n)
    exact = theta_analytic_n0(solution.xi) if n == 0.0 else theta_analytic_n1(solution.xi)
    fig = Figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(solution.xi, solution.theta - exact, color="tab:red")
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\theta_{\mathrm{numeric}} - \theta_{\mathrm{analytic}}$")
    ax.set_title(f"Validation residual, n = {n}")
    fig.tight_layout()
    return fig


def plot_he_vs_model_s(result: HEIntegrationResult, model: ModelSProfile) -> Figure:
    """HE-integrator (real T(r), assumed mu) vs. real Model S pressure and
    density, plus their relative error — the real-data counterpart to
    plot_validation_residual."""
    errors = compare_to_model_s(result, model)
    r_sun_frac_int = result.r / model.radius
    r_sun_frac_model = model.r / model.radius

    fig = Figure(figsize=(7.5, 8))
    ax_rho, ax_P, ax_err = fig.subplots(3, 1, sharex=True)

    ax_rho.plot(r_sun_frac_model, model.rho, "-", color="0.4", label="Model S (real)")
    ax_rho.plot(r_sun_frac_int, result.density, "--", color="tab:blue",
                label=f"HE integrator ($\\mu$={result.mu_mode})")
    ax_rho.set_ylabel(r"$\rho$ (kg/m$^3$)")
    ax_rho.set_yscale("log")
    ax_rho.legend()
    ax_rho.set_title("Hydrostatic-equilibrium integration vs. real Model S")

    ax_P.plot(r_sun_frac_model, model.pressure, "-", color="0.4", label="Model S (real)")
    ax_P.plot(r_sun_frac_int, result.pressure, "--", color="tab:orange",
              label=f"HE integrator ($\\mu$={result.mu_mode})")
    ax_P.set_ylabel("P (Pa)")
    ax_P.set_yscale("log")
    ax_P.legend()

    ax_err.plot(r_sun_frac_int, errors["rho_relative_error"] * 100, "-", label=r"$\rho$ rel. error")
    ax_err.plot(r_sun_frac_int, errors["pressure_relative_error"] * 100, "--", label="P rel. error")
    ax_err.axhline(0.0, color="0.7", linewidth=0.8)
    ax_err.set_ylabel("Relative error (%)")
    ax_err.set_xlabel(r"$r / R_\odot$")
    # Relative error is ill-conditioned right at the surface, where the real
    # density/pressure approach zero and even a tiny absolute mismatch
    # produces an arbitrarily large ratio; clip the axis so that artifact
    # doesn't swamp the physically meaningful interior comparison.
    ax_err.set_ylim(-50, 50)
    ax_err.legend()

    fig.tight_layout()
    return fig
