"""
polystar: a validated Lane-Emden polytropic stellar-structure solver.

Public API
----------
solve_lane_emden(n, config=None) -> LaneEmdenSolution
    Solve the dimensionless Lane-Emden equation for polytropic index n.

scale_to_physical(solution, rho_c, K) -> PhysicalProfile
    Convert a dimensionless solution to physical (SI) radius, density,
    pressure, and enclosed-mass profiles given a central density and
    polytropic constant.
"""

from __future__ import annotations

from .he_integrator import HEIntegrationResult, integrate_hydrostatic_equilibrium
from .lane_emden import solve_lane_emden
from .model_s_data import ModelSProfile, load_model_s
from .models import LaneEmdenSolution, PhysicalProfile, SolverConfig
from .scaling import scale_to_physical

__version__ = "1.0.0"

__all__ = [
    "HEIntegrationResult",
    "LaneEmdenSolution",
    "ModelSProfile",
    "PhysicalProfile",
    "SolverConfig",
    "integrate_hydrostatic_equilibrium",
    "load_model_s",
    "scale_to_physical",
    "solve_lane_emden",
]
