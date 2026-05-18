"""Core data structures and TSV loading helpers for source-token counts.

The JADT reproduction package represents all estimator inputs as a source-by-
token count matrix.  Rows are population units such as users, authors, or
speakers; columns are token types such as emoji or word tokens.  This module is
intentionally small so that the estimator definitions remain inspectable outside
of the larger MDLSW research codebase from which the paper originated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class CountsDataset:
    """A source-by-token count table with explicit source and vocabulary labels.

    Parameters
    ----------
    counts:
        Two-dimensional non-negative count matrix with shape ``(S, V)``.
    source_ids:
        Source identifiers corresponding to matrix rows.
    vocab:
        Token labels corresponding to matrix columns.
    metadata:
        Optional provenance/configuration metadata carried alongside the matrix.
    """

    counts: np.ndarray
    source_ids: list[str]
    vocab: list[str]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate matrix shape, labels, and non-negativity after construction."""
        matrix = np.asarray(self.counts, dtype=np.float64)

        # Validate the basic array contract before checking label alignment.
        if matrix.ndim != 2:
            raise ValueError(f"counts must be 2D, got shape {matrix.shape!r}.")
        if np.any(~np.isfinite(matrix)) or np.any(matrix < 0):
            raise ValueError("counts must contain finite non-negative values.")

        # Ensure source and vocabulary labels match the matrix dimensions.
        if matrix.shape[0] != len(self.source_ids):
            raise ValueError("number of source_ids must match count rows.")
        if matrix.shape[1] != len(self.vocab):
            raise ValueError("number of vocab labels must match count columns.")

        # Store a defensive float matrix even if the caller supplied integers.
        object.__setattr__(self, "counts", matrix)
        object.__setattr__(self, "source_ids", [str(s) for s in self.source_ids])
        object.__setattr__(self, "vocab", [str(t) for t in self.vocab])

    @property
    def S(self) -> int:
        """Return the number of sources."""
        return int(self.counts.shape[0])

    @property
    def V(self) -> int:
        """Return the number of token types."""
        return int(self.counts.shape[1])

    @property
    def source_masses(self) -> np.ndarray:
        """Return per-source token totals."""
        return self.counts.sum(axis=1).astype(np.float64)

    @property
    def token_masses(self) -> np.ndarray:
        """Return per-token corpus totals."""
        return self.counts.sum(axis=0).astype(np.float64)

    @property
    def total_mass(self) -> float:
        """Return the total token mass in the dataset."""
        return float(self.source_masses.sum())

    def require_positive_sources(self) -> None:
        """Raise when one or more source rows have no tokens."""
        zero_mask = self.source_masses <= 0

        # Zero-mass rows make source-normalized estimators undefined.
        if np.any(zero_mask):
            bad = [self.source_ids[i] for i in np.where(zero_mask)[0]]
            raise ValueError(f"all sources must have positive token mass; zero rows: {bad!r}")


def build_counts_dataset(
    rows: Sequence[tuple[str, Sequence[str]]],
    *,
    min_freq: int = 1,
    metadata: Mapping[str, object] | None = None,
) -> CountsDataset:
    """Build a :class:`CountsDataset` from ``(source_id, tokens)`` rows.

    Tokens from repeated source identifiers are aggregated.  Source order follows
    first appearance in ``rows``; vocabulary order is deterministic and sorted by
    token string after applying the global ``min_freq`` threshold.
    """
    if int(min_freq) < 1:
        raise ValueError("min_freq must be at least 1.")

    # Accumulate source order and raw token counts in plain dictionaries.
    source_order: list[str] = []
    counts_by_source: dict[str, dict[str, int]] = {}
    global_counts: dict[str, int] = {}

    for source_id, tokens in rows:
        sid = str(source_id)
        if sid not in counts_by_source:
            counts_by_source[sid] = {}
            source_order.append(sid)

        # Count non-empty token strings for this source and globally.
        for token in tokens:
            tok = str(token)
            if not tok:
                continue
            counts_by_source[sid][tok] = counts_by_source[sid].get(tok, 0) + 1
            global_counts[tok] = global_counts.get(tok, 0) + 1

    # Apply the vocabulary threshold deterministically.
    vocab = sorted(tok for tok, count in global_counts.items() if count >= int(min_freq))
    if not source_order:
        raise ValueError("cannot build a dataset from zero source rows.")
    if not vocab:
        raise ValueError("no token types remain after min_freq filtering.")

    # Fill the dense count matrix used by the small reproduction estimators.
    token_to_col = {token: idx for idx, token in enumerate(vocab)}
    matrix = np.zeros((len(source_order), len(vocab)), dtype=np.float64)
    for row_idx, sid in enumerate(source_order):
        for token, count in counts_by_source[sid].items():
            col_idx = token_to_col.get(token)
            if col_idx is not None:
                matrix[row_idx, col_idx] = float(count)

    return CountsDataset(
        counts=matrix,
        source_ids=source_order,
        vocab=vocab,
        metadata=dict(metadata or {}),
    )


def split_sequence(value: object, *, delimiter: str = "-") -> list[str]:
    """Split a delimiter-coded token sequence into non-empty tokens."""
    if pd.isna(value):
        return []

    # The final emoji TSV uses hyphen-delimited sequences, matching MDLSW.
    text = str(value)
    return [part for part in text.split(delimiter) if part]


def load_sequence_tsv(
    path: str | Path,
    *,
    source_col: str = "source_id",
    sequence_col: str = "sequence",
    delimiter: str = "-",
    sep: str = "\t",
    min_freq: int = 1,
) -> CountsDataset:
    """Load a source/sequence TSV into a :class:`CountsDataset`.

    The loader is deliberately generic.  It is suitable for the public toy emoji
    data and for the WNS-derived emoji TSV generated locally by authorized users.
    """
    path = Path(path)
    frame = pd.read_csv(path, sep=sep)

    # Fail early with useful messages when the expected columns are absent.
    missing = [col for col in (source_col, sequence_col) if col not in frame.columns]
    if missing:
        raise ValueError(f"missing required column(s) in {path}: {missing!r}")

    # Convert each row into the normalized builder representation.
    rows = [
        (str(row[source_col]), split_sequence(row[sequence_col], delimiter=delimiter))
        for _, row in frame.iterrows()
    ]
    return build_counts_dataset(
        rows,
        min_freq=min_freq,
        metadata={
            "path": str(path),
            "source_col": source_col,
            "sequence_col": sequence_col,
            "delimiter": delimiter,
            "sep": sep,
            "min_freq": int(min_freq),
        },
    )
