"""Distance and normalization utilities used by the reproduction estimators."""

from __future__ import annotations

import numpy as np


def normalize_nonnegative(values: np.ndarray, *, name: str = "values") -> np.ndarray:
    """Return a normalized copy of a finite non-negative vector."""
    vector = np.asarray(values, dtype=np.float64)

    # Validate before normalization so invalid estimates fail loudly.
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {vector.shape!r}.")
    if np.any(~np.isfinite(vector)) or np.any(vector < 0):
        raise ValueError(f"{name} must contain finite non-negative values.")

    # Convert positive mass vectors into probability distributions.
    total = float(vector.sum())
    if total <= 0:
        raise ValueError(f"{name} must have positive total mass.")
    return vector / total


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    r"""Compute total variation distance between two discrete distributions.

    The distance is
    $$\mathrm{TV}(p,q)=\frac{1}{2}\sum_i |p_i-q_i|.$$
    Inputs are defensively renormalized, allowing callers to pass non-negative
    mass vectors as long as both have positive total mass.
    """
    p_norm = normalize_nonnegative(np.asarray(p, dtype=np.float64), name="p")
    q_norm = normalize_nonnegative(np.asarray(q, dtype=np.float64), name="q")

    # Total variation is defined only on comparable supports.
    if p_norm.shape != q_norm.shape:
        raise ValueError(f"p and q must have the same shape, got {p_norm.shape} and {q_norm.shape}.")
    return 0.5 * float(np.abs(p_norm - q_norm).sum())
