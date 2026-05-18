"""Tests for source-size imbalance summaries."""

from __future__ import annotations

import pytest

from typical_source_estimation.data import build_counts_dataset
from typical_source_estimation.source_summary import summarize_source_sizes


def test_source_summary_reports_top_and_bottom_shares() -> None:
    """Source-size summaries should use the paper's top/bottom conventions."""
    ds = build_counts_dataset(
        [
            ("s1", ["x"]),
            ("s2", ["x", "x"]),
            ("s3", ["x", "x", "x"]),
            ("s4", ["x", "x", "x", "x"]),
            ("s5", ["x", "x", "x", "x", "x"]),
        ]
    )
    summary = summarize_source_sizes(ds)

    # With five sources, top 10% rounds up to one source and bottom 50% floors to two.
    assert summary.source_count == 5
    assert summary.total_tokens == 15
    assert summary.top_10pct_source_count == 1
    assert summary.top_10pct_token_share == pytest.approx(5.0 / 15.0)
    assert summary.bottom_50pct_source_count == 2
    assert summary.bottom_50pct_token_share == pytest.approx(3.0 / 15.0)
    assert summary.median_tokens_per_source == pytest.approx(3.0)
    assert summary.max_tokens_per_source == 5
