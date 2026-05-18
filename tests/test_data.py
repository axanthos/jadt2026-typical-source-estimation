"""Tests for dataset construction and TSV loading helpers."""

from __future__ import annotations

import numpy as np
import pytest

from typical_source_estimation.data import CountsDataset, build_counts_dataset, load_sequence_tsv, split_sequence


def test_counts_dataset_validates_shape_and_labels() -> None:
    """CountsDataset should reject malformed matrices and label mismatches."""

    # Non-2D matrices cannot represent source-by-token tables.
    with pytest.raises(ValueError, match="counts must be 2D"):
        CountsDataset(counts=np.array([1.0, 2.0]), source_ids=["s1"], vocab=["a", "b"])

    # Negative and non-finite entries are invalid count values.
    with pytest.raises(ValueError, match="finite non-negative"):
        CountsDataset(counts=np.array([[1.0, -1.0]]), source_ids=["s1"], vocab=["a", "b"])

    # Label lists must align exactly with the matrix dimensions.
    with pytest.raises(ValueError, match="source_ids"):
        CountsDataset(counts=np.array([[1.0, 0.0]]), source_ids=["s1", "s2"], vocab=["a", "b"])
    with pytest.raises(ValueError, match="vocab"):
        CountsDataset(counts=np.array([[1.0, 0.0]]), source_ids=["s1"], vocab=["a"])


def test_counts_dataset_exposes_mass_properties() -> None:
    """CountsDataset should expose consistent source, token, and total masses."""
    ds = CountsDataset(
        counts=np.array([[1, 2], [3, 0]]),
        source_ids=["1", "2"],
        vocab=["a", "b"],
        metadata={"kind": "toy"},
    )

    # The dataclass coerces counts to floating point and labels to strings.
    assert ds.S == 2
    assert ds.V == 2
    assert ds.counts.dtype == np.float64
    assert ds.source_ids == ["1", "2"]
    assert ds.vocab == ["a", "b"]
    assert np.allclose(ds.source_masses, np.array([3.0, 3.0]))
    assert np.allclose(ds.token_masses, np.array([4.0, 2.0]))
    assert ds.total_mass == pytest.approx(6.0)
    assert ds.metadata == {"kind": "toy"}


def test_build_counts_dataset_aggregates_sources_and_filters_vocab() -> None:
    """The builder should aggregate repeated sources and apply global min_freq."""
    ds = build_counts_dataset(
        [
            ("b", ["z", "x"]),
            ("a", ["x"]),
            ("b", ["x", "", "rare"]),
            ("c", ["rare"]),
        ],
        min_freq=2,
        metadata={"origin": "unit-test"},
    )

    # Source order follows first appearance; vocabulary is sorted after filtering.
    assert ds.source_ids == ["b", "a", "c"]
    assert ds.vocab == ["rare", "x"]
    assert np.allclose(ds.counts, np.array([[1.0, 2.0], [0.0, 1.0], [1.0, 0.0]]))
    assert ds.metadata == {"origin": "unit-test"}


def test_build_counts_dataset_rejects_empty_or_overfiltered_inputs() -> None:
    """The builder should fail early for unusable rows or vocabularies."""

    # An empty source sequence gives no row labels at all.
    with pytest.raises(ValueError, match="zero source rows"):
        build_counts_dataset([])

    # A threshold above every token frequency leaves no usable vocabulary.
    with pytest.raises(ValueError, match="no token types remain"):
        build_counts_dataset([("s1", ["a"]), ("s2", ["b"])], min_freq=2)

    # min_freq must be a positive integer threshold.
    with pytest.raises(ValueError, match="min_freq"):
        build_counts_dataset([("s1", ["a"])], min_freq=0)


def test_split_sequence_drops_empty_parts_and_missing_values() -> None:
    """Delimited sequences should be split into non-empty token strings."""
    assert split_sequence("a--b-") == ["a", "b"]
    assert split_sequence("a|b||", delimiter="|") == ["a", "b"]
    assert split_sequence(float("nan")) == []


def test_load_sequence_tsv_reads_counts_and_metadata(tmp_path) -> None:
    """The TSV loader should convert sequence rows into a CountsDataset."""
    tsv_path = tmp_path / "toy.tsv"
    tsv_path.write_text("source_id\tsequence\ns1\ta-b-b\ns2\tb\n", encoding="utf-8")

    ds = load_sequence_tsv(tsv_path)

    # The loader delegates to the same deterministic counting contract.
    assert ds.source_ids == ["s1", "s2"]
    assert ds.vocab == ["a", "b"]
    assert np.allclose(ds.counts, np.array([[1.0, 2.0], [0.0, 1.0]]))
    assert ds.metadata["path"] == str(tsv_path)
    assert ds.metadata["source_col"] == "source_id"
    assert ds.metadata["sequence_col"] == "sequence"


def test_load_sequence_tsv_reports_missing_columns(tmp_path) -> None:
    """The TSV loader should fail with an actionable missing-column error."""
    tsv_path = tmp_path / "bad.tsv"
    tsv_path.write_text("source_id\ttext\ns1\ta-b\n", encoding="utf-8")

    # This protects the reproduction scripts from silently reading the wrong table.
    with pytest.raises(ValueError, match="missing required column"):
        load_sequence_tsv(tsv_path)
