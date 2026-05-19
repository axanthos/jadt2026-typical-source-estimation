"""Tests for the lexical input comparison diagnostic script."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.compare_lexical_inputs import build_outputs


def test_compare_lexical_inputs_reports_source_and_token_deltas(tmp_path: Path) -> None:
    """Comparison outputs concrete global, source, and token deltas."""
    old_path = tmp_path / "old.tsv"
    new_path = tmp_path / "new.tsv"
    outdir = tmp_path / "compare"

    # The legacy-shaped file includes one system row and one placeholder token.
    old_path.write_text(
        "who\tstring\n"
        "#wns.system\t__GROUP_CREATION_MESSAGE__\n"
        "#wns.user.001\tBonjour merci _TOWN_NAME_\n"
        "#wns.user.002\tOui haha\n",
        encoding="utf-8",
    )
    new_path.write_text(
        "source_id\ttext\n"
        "#wns.user.001\tBonjour merci demain\n"
        "#wns.user.002\tOui\n",
        encoding="utf-8",
    )

    # Build comparison outputs with the same filter used by lexical tables.
    build_outputs(
        old_path=str(old_path),
        new_path=str(new_path),
        outdir=str(outdir),
        old_source_col="who",
        old_text_col="string",
        new_source_col="source_id",
        new_text_col="text",
        sep="\t",
        token_regex=r"(?u)\b\w+\b",
        exclude_source_regex=r"(?i)^(#?wns\.system|#?system|system)$",
        markdown_rows=10,
    )

    # The global summary sees the new filtered token count as unchanged overall.
    summary = pd.read_csv(outdir / "global_summary.tsv", sep="\t")
    lookup = dict(zip(summary["metric"], summary["delta_new_minus_old"]))
    assert lookup["input_rows"] == -1
    assert lookup["rows_after_source_filter"] == 0
    assert lookup["filtered_token_count"] == 0
    assert lookup["filtered_vocab_types"] == 0

    # Source and token deltas identify where the offset moved.
    source_deltas = pd.read_csv(outdir / "source_token_deltas.tsv", sep="\t")
    user2 = source_deltas[source_deltas["source_id"] == "#wns.user.002"].iloc[0]
    assert int(user2["filtered_delta"]) == -1

    filtered_tokens = pd.read_csv(outdir / "token_count_deltas_filtered.tsv", sep="\t")
    token_lookup = dict(zip(filtered_tokens["token"], filtered_tokens["delta"]))
    assert token_lookup["demain"] == 1
    assert token_lookup["haha"] == -1

    # Raw deltas preserve artifact reasons for terms removed downstream.
    raw_tokens = pd.read_csv(outdir / "token_count_deltas_raw.tsv", sep="\t")
    town = raw_tokens[raw_tokens["token"] == "_town_name_"].iloc[0]
    assert town["artifact_reason"] == "deidentification_placeholder"
    assert (outdir / "token_count_deltas_filtered.md").exists()
