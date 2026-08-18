"""Sanity checks on the bundled real Model S data (not synthetic)."""

import numpy as np
import pytest

from polystar.constants import M_SUN, R_SUN
from polystar.model_s_data import load_model_s


@pytest.fixture(scope="module")
def model():
    return load_model_s()


def test_radius_ascending(model):
    assert np.all(np.diff(model.r) > 0)


def test_density_and_pressure_positive(model):
    assert np.all(model.rho > 0)
    assert np.all(model.pressure > 0)
    assert np.all(model.temperature > 0)


def test_density_and_pressure_decrease_outward(model):
    # Not strictly monotonic at machine precision, but should be over any
    # reasonably coarse stride through a convective/radiative-zone model.
    stride = 50
    assert np.all(np.diff(model.rho[::stride]) <= 0)
    assert np.all(np.diff(model.pressure[::stride]) <= 0)


def test_central_density_matches_known_model_s_value(model):
    # Model S center: rho_c ~ 1.5e5 kg/m^3 (~150 g/cm^3).
    assert model.rho[0] == pytest.approx(1.5e5, rel=0.05)


def test_total_mass_matches_solar_mass(model):
    # Integrating the real density profile should recover M_sun to a few
    # percent (the file's density grid isn't dense enough for exact
    # agreement, and the model's own inner boundary starts at a finite,
    # nonzero radius, not r=0).
    assert model.enclosed_mass[-1] == pytest.approx(M_SUN, rel=0.05)


def test_radius_matches_solar_radius(model):
    assert model.radius == pytest.approx(R_SUN, rel=0.01)


def test_mean_molecular_weight_in_physical_range(model):
    # Fully ionized hydrogen/helium plasma: mu is O(0.5-1.3) throughout.
    assert np.all(model.mean_molecular_weight > 0.1)
    assert np.all(model.mean_molecular_weight < 2.0)
