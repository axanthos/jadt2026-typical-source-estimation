"""Typical-source estimation tools for the JADT 2026 reproduction package.

The package root exports dependency-light estimator primitives.  WNS preparation
helpers live in :mod:`typical_source_estimation.wns` and intentionally require
the emoji segmentation dependency used by the WNS preparation scripts.
"""

from typical_source_estimation.data import CountsDataset, build_counts_dataset, load_sequence_tsv, split_sequence
from typical_source_estimation.estimators import (
    EstimatorResult,
    aggregate_source_profiles,
    capped_mass,
    capped_mass_alpha,
    pooled_mle,
    source_profiles,
    uniform_sources,
)
from typical_source_estimation.metrics import normalize_nonnegative, total_variation
from typical_source_estimation.source_summary import SourceSizeSummary, summarize_source_sizes
from typical_source_estimation.simulation import SimulationConfig, run_coupling_simulation, summarize_simulation

__all__ = [
    "CountsDataset",
    "EstimatorResult",
    "SimulationConfig",
    "SourceSizeSummary",
    "aggregate_source_profiles",
    "build_counts_dataset",
    "capped_mass",
    "capped_mass_alpha",
    "load_sequence_tsv",
    "normalize_nonnegative",
    "pooled_mle",
    "run_coupling_simulation",
    "source_profiles",
    "split_sequence",
    "summarize_simulation",
    "summarize_source_sizes",
    "total_variation",
    "uniform_sources",
]
