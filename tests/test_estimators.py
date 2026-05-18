"""Tests for the paper's core POOL, UNIF, and CAP estimators."""

from __future__ import annotations

import numpy as np
import pytest

from typical_source_estimation.data import build_counts_dataset
from typical_source_estimation.estimators import capped_mass, capped_mass_alpha, pooled_mle, uniform_sources


def _toy_dataset():
    """Return a two-source dataset with intentionally imbalanced source sizes."""
    return build_counts_dataset(
        [
            ("small", ["a", "b"]),
            ("large", ["a", "a", "a", "a", "b", "b"]),
        ]
    )


def test_pooled_mle_matches_token_totals() -> None:
    """POOL should be the token-frequency distribution over the whole corpus."""
    ds = _toy_dataset()
    result = pooled_mle(ds)

    # Counts are a=5, b=3 over eight total tokens.
    assert np.allclose(result.q_hat, np.array([5.0 / 8.0, 3.0 / 8.0]))
    assert np.allclose(result.source_weights, np.array([2.0 / 8.0, 6.0 / 8.0]))


def test_uniform_sources_averages_source_profiles() -> None:
    """UNIF should average normalized source profiles with equal source weights."""
    ds = _toy_dataset()
    result = uniform_sources(ds)

    # Source profiles are [.5, .5] and [4/6, 2/6].
    expected = np.array([(0.5 + 4.0 / 6.0) / 2.0, (0.5 + 2.0 / 6.0) / 2.0])
    assert np.allclose(result.q_hat, expected)
    assert np.allclose(result.source_weights, np.array([0.5, 0.5]))


def test_capped_mass_uses_minimum_of_source_mass_and_cap() -> None:
    """CAP should weight source profiles by capped effective masses."""
    ds = _toy_dataset()
    result = capped_mass(ds, cap_mass=3.0)

    # Effective source masses are min(2,3)=2 and min(6,3)=3.
    expected_weights = np.array([2.0 / 5.0, 3.0 / 5.0])
    assert np.allclose(result.source_weights, expected_weights)
    assert np.allclose(result.q_hat, expected_weights @ np.array([[0.5, 0.5], [4.0 / 6.0, 2.0 / 6.0]]))


def test_capped_mass_alpha_maps_to_equal_share_cap() -> None:
    """The paper's alpha parameter should scale the equal-share mass N/S."""
    ds = _toy_dataset()
    result = capped_mass_alpha(ds, alpha=1.0)

    # Equal-share mass is 8/2=4; effective source masses are 2 and 4.
    assert result.metadata["cap_mass"] == pytest.approx(4.0)
    assert np.allclose(result.source_weights, np.array([2.0 / 6.0, 4.0 / 6.0]))


def test_estimators_sum_to_one() -> None:
    """All estimator outputs should be valid probability distributions."""
    ds = _toy_dataset()

    # The three public estimators should all return normalized q and weights.
    for result in [pooled_mle(ds), uniform_sources(ds), capped_mass_alpha(ds, alpha=1.0)]:
        assert float(result.q_hat.sum()) == pytest.approx(1.0)
        assert float(result.source_weights.sum()) == pytest.approx(1.0)
        assert np.all(result.q_hat >= 0)
        assert np.all(result.source_weights >= 0)
