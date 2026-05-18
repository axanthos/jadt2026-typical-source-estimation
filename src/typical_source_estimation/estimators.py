"""Population estimators for the JADT typical-source reproduction package.

The estimators in this module are intentionally limited to the paper's core
comparison: pooled token weighting (``POOL``), uniform source weighting
(``UNIF``), and capped source-mass weighting (``CAP``).  Additional
experimental estimators from earlier exploratory work are deliberately excluded
from this paper-specific reproduction package.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from typical_source_estimation.data import CountsDataset
from typical_source_estimation.metrics import normalize_nonnegative


@dataclass(frozen=True, slots=True)
class EstimatorResult:
    """Result of a population estimator.

    Attributes
    ----------
    q_hat:
        Estimated population distribution over token types.
    source_weights:
        Source weights used to aggregate source profiles.
    metadata:
        Estimator parameters and compact diagnostic values.
    """

    q_hat: np.ndarray
    source_weights: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize and validate estimator outputs after construction."""
        q = normalize_nonnegative(np.asarray(self.q_hat, dtype=np.float64), name="q_hat")
        w = normalize_nonnegative(np.asarray(self.source_weights, dtype=np.float64), name="source_weights")

        # Freeze normalized arrays into the dataclass so downstream code is simple.
        object.__setattr__(self, "q_hat", q)
        object.__setattr__(self, "source_weights", w)


def source_profiles(dataset: CountsDataset) -> np.ndarray:
    """Return empirical token distributions for each source."""
    dataset.require_positive_sources()
    masses = dataset.source_masses

    # Divide each source row by its own mass to obtain source profiles.
    return dataset.counts / masses[:, None]


def aggregate_source_profiles(dataset: CountsDataset, weights: np.ndarray) -> np.ndarray:
    """Aggregate source profiles with a source-weight vector."""
    w = normalize_nonnegative(weights, name="weights")
    if w.shape[0] != dataset.S:
        raise ValueError(f"weights length must match sources: {w.shape[0]} != {dataset.S}.")

    # Source profiles are normalized before weighted averaging.
    profiles = source_profiles(dataset)
    return normalize_nonnegative(w @ profiles, name="q_hat")


def pooled_mle(dataset: CountsDataset) -> EstimatorResult:
    """Estimate the population by pooling all source tokens."""
    dataset.require_positive_sources()
    token_totals = dataset.token_masses
    source_weights = normalize_nonnegative(dataset.source_masses, name="source_masses")

    # Pooled MLE is equivalent to source-profile averaging by token mass.
    q_hat = normalize_nonnegative(token_totals, name="token_totals")
    return EstimatorResult(
        q_hat=q_hat,
        source_weights=source_weights,
        metadata={"estimator": "POOL"},
    )


def uniform_sources(dataset: CountsDataset) -> EstimatorResult:
    """Estimate the population by giving each source equal weight."""
    dataset.require_positive_sources()

    # Uniform source weighting implements the typical-source endpoint.
    weights = np.full(dataset.S, 1.0 / float(dataset.S), dtype=np.float64)
    q_hat = aggregate_source_profiles(dataset, weights)
    return EstimatorResult(
        q_hat=q_hat,
        source_weights=weights,
        metadata={"estimator": "UNIF"},
    )


def capped_mass(dataset: CountsDataset, *, cap_mass: float) -> EstimatorResult:
    """Estimate the population using capped source masses.

    Each source receives effective mass ``min(T_s, cap_mass)``.  The resulting
    effective masses are normalized to source weights and used to average source
    profiles.  This is the paper's conservative compromise between token-mass
    weighting and uniform source weighting.
    """
    dataset.require_positive_sources()
    if not np.isfinite(cap_mass) or float(cap_mass) <= 0:
        raise ValueError("cap_mass must be a positive finite number.")

    # Cap each source's contribution before normalizing source weights.
    masses = dataset.source_masses
    effective_masses = np.minimum(masses, float(cap_mass))
    weights = normalize_nonnegative(effective_masses, name="effective_masses")

    # Aggregate source profiles using the capped weights.
    q_hat = aggregate_source_profiles(dataset, weights)
    return EstimatorResult(
        q_hat=q_hat,
        source_weights=weights,
        metadata={
            "estimator": "CAP",
            "cap_mass": float(cap_mass),
        },
    )


def capped_mass_alpha(dataset: CountsDataset, *, alpha: float) -> EstimatorResult:
    r"""Estimate the population with cap ``alpha * (N/S)``.

    This is the JADT paper parameterization of capped weighting.  The baseline
    cap is the equal-share source mass $N/S$ and ``alpha`` scales it
    multiplicatively.
    """
    if not np.isfinite(alpha) or float(alpha) <= 0:
        raise ValueError("alpha must be a positive finite number.")
    if dataset.S <= 0:
        raise ValueError("dataset must contain at least one source.")

    # Convert the paper's alpha parameter into an absolute cap mass.
    cap = float(alpha) * (dataset.total_mass / float(dataset.S))
    result = capped_mass(dataset, cap_mass=cap)
    return EstimatorResult(
        q_hat=result.q_hat,
        source_weights=result.source_weights,
        metadata={
            **result.metadata,
            "alpha": float(alpha),
            "equal_share_mass": dataset.total_mass / float(dataset.S),
        },
    )
