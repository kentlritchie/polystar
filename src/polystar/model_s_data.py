"""
Loader for "Model S" (Christensen-Dalsgaard et al. 1996, Science 272, 1286),
the standard helioseismically-calibrated solar structure model.

Source: J. Christensen-Dalsgaard's solar-models page,
https://users-phys.au.dk/jcd/solar_models/ , file `cptrho.l5bi.d.15c`
("Limited set of variables for Model S": sound speed, density, pressure,
adiabatic exponent Gamma_1, and temperature vs. r/R). The file is bundled
under polystar/data/model_s_cptrho.dat and distributed under the same terms
as the original (freely available for scientific/educational use; see the
reference above). It is real, published data, not a synthetic stand-in.

This module is the "real data" counterpart to the purely theoretical
Lane-Emden solver: it is used both as a standalone dataset (real solar
structure) and to drive he_integrator.py's non-polytropic hydrostatic
integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from .constants import (
    DYN_PER_CM2_TO_PA,
    G_PER_CM3_TO_KG_PER_M3,
    K_B,
    M_H,
    R_SUN,
)

_DATA_FILE = "model_s_cptrho.dat"


@dataclass(frozen=True)
class ModelSProfile:
    """Real Model S structure, ascending in radius, converted to SI units."""

    r: np.ndarray  # m
    rho: np.ndarray  # kg/m^3
    pressure: np.ndarray  # Pa
    temperature: np.ndarray  # K
    gamma1: np.ndarray  # dimensionless
    enclosed_mass: np.ndarray  # kg, integrated from rho(r)
    mean_molecular_weight: np.ndarray  # dimensionless, implied by ideal gas law
    radius: float  # m, outermost tabulated point
    source: str = "Christensen-Dalsgaard et al. (1996), Science 272, 1286 (Model S)"

    def rho_of_r(self) -> interp1d:
        return interp1d(self.r, self.rho, bounds_error=False, fill_value=(self.rho[0], 0.0))

    def pressure_of_r(self) -> interp1d:
        return interp1d(self.r, self.pressure, bounds_error=False, fill_value=(self.pressure[0], 0.0))

    def temperature_of_r(self) -> interp1d:
        return interp1d(
            self.r, self.temperature, bounds_error=False,
            fill_value=(self.temperature[0], self.temperature[-1]),
        )

    def mu_of_r(self) -> interp1d:
        return interp1d(
            self.r, self.mean_molecular_weight, bounds_error=False,
            fill_value=(self.mean_molecular_weight[0], self.mean_molecular_weight[-1]),
        )

    def mass_of_r(self) -> interp1d:
        return interp1d(
            self.r, self.enclosed_mass, bounds_error=False,
            fill_value=(0.0, self.enclosed_mass[-1]),
        )


def load_model_s() -> ModelSProfile:
    """Parse the bundled Model S data file into SI-unit arrays.

    The raw file is ordered from the surface inward (r/R descending from
    ~1.0007 to 0); this returns it re-ordered from the center outward
    (ascending r), which is what the HE integrator and plotting code want.
    """
    data_path = resources.files("polystar.data").joinpath(_DATA_FILE)
    with resources.as_file(data_path) as path:
        raw = np.loadtxt(path, comments="#")

    r_over_R, _c, rho_cgs, p_cgs, gamma1, temperature = raw.T

    order = np.argsort(r_over_R)
    r_over_R = r_over_R[order]
    rho_cgs = rho_cgs[order]
    p_cgs = p_cgs[order]
    gamma1 = gamma1[order]
    temperature = temperature[order]

    r = r_over_R * R_SUN
    rho = rho_cgs * G_PER_CM3_TO_KG_PER_M3
    pressure = p_cgs * DYN_PER_CM2_TO_PA

    enclosed_mass = np.concatenate((
        [0.0],
        cumulative_trapezoid(4.0 * np.pi * r**2 * rho, r),
    ))

    # Local mean molecular weight implied by the ideal-gas law, using the
    # model's own real P, rho, T together: mu = rho*k_B*T / (P*m_H). This
    # departs from a true ideal gas in partial-ionization zones near the
    # surface (where Gamma_1 also deviates furthest from 5/3), which is
    # visible in the profile rather than hidden.
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(pressure > 0, rho * K_B * temperature / (pressure * M_H), np.nan)
    mu[0] = mu[1]  # center: P->0 and rho->finite makes the ratio ill-conditioned

    return ModelSProfile(
        r=r,
        rho=rho,
        pressure=pressure,
        temperature=temperature,
        gamma1=gamma1,
        enclosed_mass=enclosed_mass,
        mean_molecular_weight=mu,
        radius=float(r[-1]),
    )
