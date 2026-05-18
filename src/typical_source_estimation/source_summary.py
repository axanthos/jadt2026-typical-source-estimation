"""Source-size imbalance summaries used in the JADT real-data reporting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from typical_source_estimation.data import CountsDataset


@dataclass(frozen=True, slots=True)
class SourceSizeSummary:
    """Compact diagnostics for source-size imbalance."""

    source_count: int
    total_tokens: int
    top_10pct_source_count: int
    top_10pct_token_share: float
    bottom_50pct_source_count: int
    bottom_50pct_token_share: float
    median_tokens_per_source: float
    p05_tokens_per_source: float
    max_tokens_per_source: int

    def as_dict(self) -> dict[str, int | float]:
        """Return the summary as a plain dictionary for table writing."""
        return {
            "source_count": self.source_count,
            "total_tokens": self.total_tokens,
            "top_10pct_source_count": self.top_10pct_source_count,
            "top_10pct_token_share": self.top_10pct_token_share,
            "bottom_50pct_source_count": self.bottom_50pct_source_count,
            "bottom_50pct_token_share": self.bottom_50pct_token_share,
            "median_tokens_per_source": self.median_tokens_per_source,
            "p05_tokens_per_source": self.p05_tokens_per_source,
            "max_tokens_per_source": self.max_tokens_per_source,
        }


def summarize_source_sizes(dataset: CountsDataset) -> SourceSizeSummary:
    """Compute source-size imbalance diagnostics for positive-mass sources."""
    masses = np.asarray(dataset.source_masses, dtype=np.float64)
    positive_masses = masses[masses > 0]

    # Summaries are defined on observed positive-mass sources.
    if positive_masses.size == 0:
        raise ValueError("cannot summarize source sizes with no positive-mass sources.")

    # Sort in both directions to compute top and bottom source shares.
    sorted_desc = np.sort(positive_masses)[::-1]
    sorted_asc = np.sort(positive_masses)
    source_count = int(positive_masses.size)
    total = float(positive_masses.sum())

    # Match the JADT reporting convention used in the camera-ready paper.
    top_n = max(1, int(np.ceil(0.10 * source_count)))
    bottom_n = max(1, int(np.floor(0.50 * source_count)))

    return SourceSizeSummary(
        source_count=source_count,
        total_tokens=int(round(total)),
        top_10pct_source_count=top_n,
        top_10pct_token_share=float(sorted_desc[:top_n].sum() / total),
        bottom_50pct_source_count=bottom_n,
        bottom_50pct_token_share=float(sorted_asc[:bottom_n].sum() / total),
        median_tokens_per_source=float(np.median(positive_masses)),
        p05_tokens_per_source=float(np.percentile(positive_masses, 5)),
        max_tokens_per_source=int(round(float(sorted_desc[0]))),
    )
