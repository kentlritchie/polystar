"""
Interactive Streamlit application for polystar (SRS section 9.3).

Calls only the public polystar API — no physics or numerical logic is
reimplemented here (AR-004).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow running via `streamlit run app/streamlit_app.py` from a source
# checkout without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from polystar.constants import M_SUN, R_SUN
from polystar.diagnostics import validate_analytic, validate_reference
from polystar.he_integrator import compare_to_model_s, integrate_hydrostatic_equilibrium
from polystar.lane_emden import LaneEmdenError, solve_lane_emden
from polystar.model_s_data import load_model_s
from polystar.models import SolverConfig
from polystar.plotting import (
    plot_central_concentration,
    plot_comparison,
    plot_density_profile,
    plot_he_vs_model_s,
    plot_mass_profile,
    plot_theta,
    plot_validation_residual,
)
from polystar.scaling import ScalingError, scale_to_physical

st.set_page_config(
    page_title="polystar — Lane-Emden Polytropic Star Simulator",
    layout="wide",
    page_icon="⭐",
)


@st.cache_data(show_spinner=False)
def cached_solve(n: float, xi_max: float, rtol: float, atol: float):
    config = SolverConfig(xi_max=xi_max, rtol=rtol, atol=atol)
    return solve_lane_emden(n, config=config)


@st.cache_resource(show_spinner=False)
def cached_model_s():
    return load_model_s()


@st.cache_data(show_spinner=False)
def cached_he_integration(mu_mode: str, mu_constant: float):
    model = cached_model_s()
    return integrate_hydrostatic_equilibrium(model, mu_mode=mu_mode, mu_constant=mu_constant)


st.title("⭐ polystar — Lane-Emden Polytropic Star Simulator")
st.caption(
    "Solves the Lane-Emden equation for polytropic stellar models, validates it against "
    "analytic and reference solutions, and compares a real hydrostatic-equilibrium "
    "integration against the actual Sun (Model S)."
)

with st.sidebar:
    st.header("Model parameters")
    n = st.slider("Polytropic index n", min_value=0.0, max_value=5.0, value=1.5, step=0.1)

    tolerance_preset = st.selectbox("Solver tolerance", ["default (tight)", "loose (fast)"])
    rtol, atol = (1e-10, 1e-12) if tolerance_preset == "default (tight)" else (1e-6, 1e-8)

    xi_max = st.number_input(
        "Maximum ξ (integration domain)", min_value=5.0, max_value=200.0, value=50.0, step=5.0,
        help="Increase for n approaching 5, where the surface (if any) lies far out.",
    )

    st.divider()
    st.header("Physical scaling")
    use_physical = st.toggle("Enable physical scaling", value=False)
    rho_c = K = None
    if use_physical:
        rho_c = st.number_input("Central density ρ_c (kg/m³)", min_value=1.0, value=1.5e5, format="%.3e")
        K = st.number_input("Polytropic constant K (SI)", min_value=1e-20, value=3.8e9, format="%.3e")
        show_solar_units = st.checkbox("Show physical results in solar units (R☉, M☉)", value=True)

try:
    solution = cached_solve(n, xi_max, rtol, atol)
except LaneEmdenError as exc:
    st.error(str(exc))
    st.stop()

physical_profile = None
scaling_warning = None
if use_physical:
    try:
        physical_profile = scale_to_physical(solution, rho_c, K)
    except ScalingError as exc:
        scaling_warning = str(exc)

st.subheader("Model summary")
cols = st.columns(5)
cols[0].metric("n", f"{n:g}")
cols[1].metric("ξ₁ (surface)", f"{solution.xi_surface:.4f}" if solution.xi_surface else "not found")
cols[2].metric(
    "Surface mass coeff. −ξ₁²θ′(ξ₁)",
    f"{solution.surface_mass_coefficient:.4f}" if solution.surface_mass_coefficient else "—",
)
cc = solution.central_concentration
cols[3].metric("ρ_c / ⟨ρ⟩", f"{cc:.3f}" if cc else "—")

if physical_profile is not None:
    if show_solar_units:
        cols[4].metric("R, M", f"{physical_profile.radius / R_SUN:.3f} R☉, {physical_profile.total_mass / M_SUN:.3f} M☉")
    else:
        cols[4].metric("R, M", f"{physical_profile.radius:.3e} m, {physical_profile.total_mass:.3e} kg")
elif scaling_warning:
    cols[4].warning(scaling_warning, icon="⚠️")
else:
    cols[4].metric("R, M", "scaling off")

if solution.metadata.get("warning"):
    st.warning(solution.metadata["warning"], icon="⚠️")

tab_structure, tab_comparison, tab_validation, tab_he, tab_data = st.tabs(
    ["Structure", "Comparison", "Validation", "HE vs. real Sun (Model S)", "Raw data"]
)

with tab_structure:
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_theta(solution), clear_figure=True)
    with c2:
        st.pyplot(plot_density_profile(solution), clear_figure=True)
    st.pyplot(plot_mass_profile(solution), clear_figure=True)

    if physical_profile is not None:
        st.subheader("Physical profile")
        import pandas as pd

        df_phys = pd.DataFrame({
            "r (m)": physical_profile.r,
            "rho (kg/m^3)": physical_profile.rho,
            "P (Pa)": physical_profile.pressure,
            "M(r) (kg)": physical_profile.enclosed_mass,
        })
        st.line_chart(df_phys.set_index("r (m)")[["rho (kg/m^3)"]])

with tab_comparison:
    st.write("Overlay several polytropic indices on a common normalized-radius axis.")
    n_choices = st.multiselect(
        "n values to compare", options=[0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 4.5],
        default=[0.0, 1.0, 1.5, 3.0, 4.5],
    )
    if n_choices:
        solutions = [cached_solve(nc, xi_max, rtol, atol) for nc in n_choices]
        st.pyplot(plot_comparison(solutions), clear_figure=True)
    st.pyplot(plot_central_concentration([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]), clear_figure=True)

with tab_validation:
    st.write(
        "Automated comparison against closed-form analytic solutions (n=0, n=1) and "
        "trusted tabulated reference values (n=1.5, n=3)."
    )
    for label, fn, arg in [
        ("n = 0 (analytic: θ = 1 − ξ²/6)", validate_analytic, 0.0),
        ("n = 1 (analytic: θ = sin ξ / ξ)", validate_analytic, 1.0),
        ("n = 1.5 (reference)", validate_reference, 1.5),
        ("n = 3 (reference)", validate_reference, 3.0),
    ]:
        result = fn(arg)
        icon = "✅" if result.passed else "❌"
        with st.expander(f"{icon} {label}"):
            st.json(result.details)

    st.subheader("Residual plots")
    rc1, rc2 = st.columns(2)
    rc1.pyplot(plot_validation_residual(0.0), clear_figure=True)
    rc2.pyplot(plot_validation_residual(1.0), clear_figure=True)

with tab_he:
    st.write(
        "A second, independent physics path: instead of assuming a polytropic P–ρ "
        "relation, this integrates hydrostatic equilibrium and mass continuity directly, "
        "closed with the ideal-gas law and the **real** temperature profile from "
        "[Model S](https://users-phys.au.dk/jcd/solar_models/) "
        "(Christensen-Dalsgaard et al. 1996) — an actual helioseismically-calibrated "
        "solar model, not a synthetic stand-in."
    )
    mu_mode = st.radio(
        "Mean molecular weight μ(r)", ["constant", "model_s"],
        format_func=lambda m: (
            "Constant μ ≈ 0.62 (independent assumption)"
            if m == "constant" else
            "μ(r) derived from Model S's own P, ρ, T (reproduces Model S almost exactly — a solver sanity check)"
        ),
    )
    mu_constant = 0.62
    if mu_mode == "constant":
        mu_constant = st.slider("μ (constant)", min_value=0.3, max_value=1.3, value=0.62, step=0.01)

    he_result = cached_he_integration(mu_mode, mu_constant)
    model_s = cached_model_s()
    st.pyplot(plot_he_vs_model_s(he_result, model_s), clear_figure=True)

    errors = compare_to_model_s(he_result, model_s)
    import numpy as np

    ec1, ec2 = st.columns(2)
    ec1.metric("Median |ρ error| vs. Model S", f"{np.nanmedian(np.abs(errors['rho_relative_error'])) * 100:.2f}%")
    ec2.metric("Median |P error| vs. Model S", f"{np.nanmedian(np.abs(errors['pressure_relative_error'])) * 100:.2f}%")

with tab_data:
    import pandas as pd

    end = solution.xi_surface if solution.xi_surface is not None else solution.xi[-1]
    df = pd.DataFrame({
        "xi": solution.xi,
        "theta": solution.theta,
        "dtheta_dxi": solution.dtheta_dxi,
        "rho_over_rho_c": solution.rho_normalized,
        "m_dimensionless": solution.mass_dimensionless,
        "r_over_R": solution.xi / end,
    })
    st.dataframe(df, use_container_width=True, height=400)
    st.download_button(
        "Download this profile as CSV", df.to_csv(index=False),
        file_name=f"lane_emden_n{n}.csv", mime="text/csv",
    )
