"""Tests for paper-facing reproduction table scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.reproduce_emoji_table import build_outputs as build_emoji_outputs
from scripts.reproduce_lexical_tables import build_outputs as build_lexical_outputs


def test_reproduce_emoji_table_writes_expected_outputs(tmp_path: Path) -> None:
    """Emoji reproduction writes TSV, Markdown, LaTeX, and source summaries."""
    input_path = Path("data/toy/toy_emoji.tsv")
    outdir = tmp_path / "emoji"

    # Build a compact Top-K table from the bundled toy emoji data.
    build_emoji_outputs(
        input_path=str(input_path),
        outdir=str(outdir),
        source_col="source_id",
        sequence_col="sequence",
        sep="\t",
        delimiter="-",
        min_freq=1,
        k=2,
    )

    # Check the files that define the public reproduction contract.
    topk = pd.read_csv(outdir / "emoji_topk_table.tsv", sep="\t")
    assert list(topk.columns) == ["rank", "pooled", "unif_src", "cap_a0.5", "cap_a1", "cap_a2"]
    assert topk.iloc[0]["pooled"].startswith("❤") or topk.iloc[0]["pooled"].startswith("😂")
    assert (outdir / "emoji_topk_table.md").exists()
    assert "\\twemoji" in (outdir / "emoji_topk_table.tex").read_text(encoding="utf-8")
    assert (outdir / "emoji_source_size_summary.tsv").exists()


def test_reproduce_lexical_tables_writes_tv_and_shift_tables(tmp_path: Path) -> None:
    """Lexical reproduction writes the compact TV matrix and shift outputs."""
    input_path = Path("data/toy/toy_lexical.tsv")
    outdir = tmp_path / "lexical"

    # Build lexical outputs from the bundled toy text data.
    build_lexical_outputs(
        input_path=str(input_path),
        outdir=str(outdir),
        source_col="source_id",
        text_col="text",
        sep="\t",
        min_freq=1,
        token_regex=r"(?u)\b\w+\b",
        exclude_source_regex=r"(?i)^(#?wns\.system|#?system|system)$",
        contrast="cap_a1",
        k=2,
    )

    # Verify the camera-ready matrix order and LaTeX table labels.
    tv = pd.read_csv(outdir / "lexical_tv_table.tsv", sep="\t")
    assert list(tv["Unnamed: 0"]) == ["POOL", "CAP", "UNIF"]
    assert list(tv.columns) == ["Unnamed: 0", "POOL", "CAP", "UNIF"]
    latex = (outdir / "lexical_tables.tex").read_text(encoding="utf-8")
    assert "Total variation distances" in latex
    assert "Largest signed lexical shifts" in latex
    assert "\\label{tab:lexical-tv}" in latex
    assert (outdir / "lexical_full_shift_table.tsv").exists()
