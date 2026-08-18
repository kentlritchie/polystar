"""
Physical constants, in SI units, with source attribution.

Values are CODATA 2018 (matching astropy.constants / scipy.constants), so
results are reproducible even in environments without astropy installed.
"""

from __future__ import annotations

# Newtonian constant of gravitation [m^3 kg^-1 s^-2]
# CODATA 2018 (scipy.constants.G == 6.6743e-11)
G: float = 6.6743e-11

# Boltzmann constant [J K^-1] (CODATA 2018, exact by SI definition)
K_B: float = 1.380649e-23

# Atomic mass unit / hydrogen-mass proxy [kg] (CODATA 2018)
M_H: float = 1.66053906660e-27

# Solar mass [kg] (IAU 2015 nominal value, astropy.constants.M_sun)
M_SUN: float = 1.98892e30

# Solar radius [m] (IAU 2015 nominal value, astropy.constants.R_sun)
R_SUN: float = 6.957e8

# cgs <-> SI conversions used when reading Model S (which is tabulated in cgs)
CM_TO_M: float = 1e-2
G_PER_CM3_TO_KG_PER_M3: float = 1e3
DYN_PER_CM2_TO_PA: float = 1e-1
