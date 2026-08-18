"""VAL-001..005, TEST-004: end-to-end comparison against known solutions."""

import math

import pytest

from polystar.diagnostics import validate_analytic, validate_reference


def test_n0_matches_analytic():
    result = validate_analytic(0.0)
    assert result.passed, result.details
    assert result.details["xi1_rel_error"] < 1e-6
    assert math.isclose(result.details["xi1_expected"], math.sqrt(6.0))


def test_n1_matches_analytic():
    result = validate_analytic(1.0)
    assert result.passed, result.details
    assert result.details["xi1_rel_error"] < 1e-6
    assert math.isclose(result.details["xi1_expected"], math.pi)


def test_n1p5_matches_reference():
    result = validate_reference(1.5)
    assert result.passed, result.details


def test_n3_matches_reference():
    result = validate_reference(3.0)
    assert result.passed, result.details


def test_invalid_analytic_reference_raises():
    with pytest.raises(ValueError):
        validate_analytic(2.5)


def test_invalid_reference_raises():
    with pytest.raises(ValueError):
        validate_reference(2.5)
