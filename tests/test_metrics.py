"""Tests for metric and normalization helpers."""

from __future__ import annotations

import numpy as np
import pytest

from typical_source_estimation.metrics import normalize_nonnegative, total_variation


def test_normalize_nonnegative_returns_probability_vector() -> None:
    """Positive finite vectors should be normalized to sum to one."""
    result = normalize_nonnegative(np.array([2.0, 3.0]))

    # The helper should preserve proportions while normalizing mass.
    assert np.allclose(result, np.array([0.4, 0.6]))
    assert float(result.sum()) == pytest.approx(1.0)


def test_total_variation_is_symmetric_and_renormalizes_mass_vectors() -> None:
    """TV should be symmetric and robust to positive scalar mass."""
    p = np.array([2.0, 0.0])
    q = np.array([1.0, 1.0])

    # After normalization, the vectors are [1, 0] and [.5, .5].
    assert total_variation(p, q) == pytest.approx(0.5)
    assert total_variation(q, p) == pytest.approx(0.5)


def test_total_variation_rejects_shape_mismatch() -> None:
    """TV cannot compare distributions on different supports."""
    with pytest.raises(ValueError, match="same shape"):
        total_variation(np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
