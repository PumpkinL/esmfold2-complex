"""Tests for :mod:`esmfold2_complex.contracts` metric helpers.

The metric thresholds and formatting are referenced by the report
generator and the CSV writer. The thresholds appear in user-facing
strings, so changing them is a user-visible behavior change; these
tests pin them in place.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from esmfold2_complex.contracts import (
    PLDDT_COLORS,
    classify_iptm,
    classify_pae,
    classify_plddt,
    mean_optional,
    optional_metric,
    plddt_hex,
    to_optional_float,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "NA"),
        (0.157, "0.157"),
        (1.0, "1.000"),
        (0.123456, "0.123"),
    ],
)
def test_optional_metric_formats_none_and_floats(value, expected) -> None:
    assert optional_metric(value) == expected


def test_optional_metric_honors_custom_digits() -> None:
    assert optional_metric(0.123456, digits=2) == "0.12"


def test_mean_optional_returns_none_for_empty_list() -> None:
    assert mean_optional([]) is None


def test_mean_optional_computes_arithmetic_mean() -> None:
    assert mean_optional([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)


def test_mean_optional_accepts_numpy_array() -> None:
    assert mean_optional([np.float64(2.0), np.float64(4.0)]) == pytest.approx(3.0)


def test_to_optional_float_returns_none_for_none() -> None:
    assert to_optional_float(None) is None


def test_to_optional_float_converts_python_float() -> None:
    assert to_optional_float(0.5) == 0.5
    assert isinstance(to_optional_float(0.5), float)


def test_to_optional_float_converts_numpy_scalar() -> None:
    result = to_optional_float(np.float32(0.25))
    assert result == pytest.approx(0.25, abs=1e-6)


@pytest.mark.parametrize(
    "score, expected_class",
    [
        (95.0, "very high"),
        (90.0, "very high"),
        (89.999, "good"),
        (70.0, "good"),
        (69.999, "mixed"),
        (50.0, "mixed"),
        (49.999, "low"),
        (10.0, "low"),
    ],
)
def test_classify_plddt_boundaries(score, expected_class) -> None:
    assert classify_plddt(score) == expected_class


@pytest.mark.parametrize(
    "iptm, expected_class",
    [
        (None, "unknown"),
        (0.85, "strong"),
        (0.80, "strong"),
        (0.799, "moderate"),
        (0.60, "moderate"),
        (0.599, "tentative"),
        (0.40, "tentative"),
        (0.399, "weak"),
        (0.0, "weak"),
    ],
)
def test_classify_iptm_boundaries(iptm, expected_class) -> None:
    assert classify_iptm(iptm) == expected_class


@pytest.mark.parametrize(
    "pae, expected_class",
    [
        (None, "unknown"),
        (3.0, "strong"),
        (5.0, "strong"),
        (5.001, "good"),
        (10.0, "good"),
        (10.001, "uncertain"),
        (20.0, "uncertain"),
        (20.001, "poor"),
        (100.0, "poor"),
    ],
)
def test_classify_pae_boundaries(pae, expected_class) -> None:
    assert classify_pae(pae) == expected_class


@pytest.mark.parametrize(
    "score, expected_color",
    [
        (95.0, PLDDT_COLORS["very_high"]),
        (75.0, PLDDT_COLORS["confident"]),
        (60.0, PLDDT_COLORS["low"]),
        (30.0, PLDDT_COLORS["very_low"]),
    ],
)
def test_plddt_hex_maps_to_color_bucket(score, expected_color) -> None:
    assert plddt_hex(score) == expected_color


def test_plddt_hex_returns_valid_hex_string() -> None:
    color = plddt_hex(72.0)
    assert color.startswith("#")
    assert len(color) == 7
    int(color[1:], 16)  # raises if not valid hex


def test_to_optional_float_handles_zero() -> None:
    result = to_optional_float(0)
    assert result == 0.0
    assert math.isclose(result, 0.0)
