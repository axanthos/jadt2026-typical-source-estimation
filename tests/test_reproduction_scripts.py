"""Tests for paper-facing reproduction table scripts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.reproduce_emoji_table import build_outputs as build_emoji_outputs
from scripts.reproduce_lexical_tables import build_outputs as build_lexical_outputs


ROOT = Path(__file__).resolve().parents[1]


def _run_script(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a reproduction CLI from the source checkout under test."""
    env = os.environ.copy()

    # Direct script execution from a fresh checkout needs the package source path.
    existing = env.get("PYTHONPATH", "")
    source_path = str(ROOT / "src")
    env["PYTHONPATH"] = source_path if not existing else source_path + os.pathsep + existing
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def _summary_lookup(path: Path) -> dict[str, str]:
    """Return source-size summary values keyed by metric name."""
    frame = pd.read_csv(path, sep="\t")
    return dict(zip(frame["metric"], frame["value"]))


def test_reproduce_emoji_table_writes_exact_toy_outputs(tmp_path: Path) -> None:
    """Emoji reproduction writes exact toy probabilities and summaries."""
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

    # Verify the complete toy Top-K display table, not just file existence.
    topk = pd.read_csv(outdir / "emoji_topk_table.tsv", sep="	")
    assert list(topk.columns) == ["rank", "unif_src", "cap_a0.5", "cap_a1", "cap_a2", "pooled"]
    assert topk.to_dict("records") == [
        {"rank": "1", "unif_src": "❤ 0.389", "cap_a0.5": "❤ 0.389", "cap_a1": "❤ 0.375", "cap_a2": "❤ 0.333", "pooled": "❤ 0.333"},
        {"rank": "2", "unif_src": "👍 0.333", "cap_a0.5": "👍 0.333", "cap_a1": "👍 0.312", "cap_a2": "👍 0.333", "pooled": "👍 0.333"},
        {"rank": "Head mass", "unif_src": "0.722", "cap_a0.5": "0.722", "cap_a1": "0.688", "cap_a2": "0.667", "pooled": "0.667"},
        {"rank": "TV to POOL", "unif_src": "0.056", "cap_a0.5": "0.056", "cap_a1": "0.042", "cap_a2": "0.000", "pooled": "--"},
        {"rank": "TV to UNIF", "unif_src": "--", "cap_a0.5": "0.000", "cap_a1": "0.035", "cap_a2": "0.056", "pooled": "0.056"},
    ]

    # Confirm Markdown and LaTeX outputs include the generated table content.
    markdown = (outdir / "emoji_topk_table.md").read_text(encoding="utf-8")
    assert "| rank" in markdown
    assert "❤ 0.333" in markdown
    latex = (outdir / "emoji_topk_table.tex").read_text(encoding="utf-8")
    assert r"\multicolumn{3}{c}{\textsc{cap}}" in latex
    assert r"TV to \textsc{pool}" in latex
    assert r"\twemoji" in latex

    # Source-size summaries should use the same toy source masses exactly.
    summary_lookup = _summary_lookup(outdir / "emoji_source_size_summary.tsv")
    assert summary_lookup == {
        "sources": "3",
        "tokens": "9",
        "top 10% token share": "44.4% (1 sources)",
        "bottom 50% token share": "22.2% (1 sources)",
        "median tokens/source": "3",
        "5th percentile tokens/source": "2",
        "max tokens/source": "4",
    }


def test_reproduce_lexical_tables_writes_exact_toy_outputs(tmp_path: Path) -> None:
    """Lexical reproduction writes exact toy TV, shifts, and summaries."""
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

    # Verify the camera-ready matrix order and selected numeric values.
    tv = pd.read_csv(outdir / "lexical_tv_table.tsv", sep="\t")
    assert tv.to_dict("records") == [
        {"Unnamed: 0": "POOL", "POOL": "--", "CAP": "0.015", "UNIF": "0.056"},
        {"Unnamed: 0": "CAP", "POOL": "0.015", "CAP": "--", "UNIF": "0.040"},
        {"Unnamed: 0": "UNIF", "POOL": "0.056", "CAP": "0.040", "UNIF": "--"},
    ]

    # Check that the signed-shift table reflects the toy estimator contrast.
    shifts = pd.read_csv(outdir / "lexical_shift_table.tsv", sep="\t")
    assert list(shifts["higher_under_cap"]) == ["bonjour", "merci"]
    assert list(shifts["higher_under_pool"]) == ["haha", "oui"]
    assert list(shifts["cap_source_count"]) == [1, 2]
    assert list(shifts["pool_source_count"]) == [1, 2]
    assert float(shifts.iloc[0]["cap_delta_per_10k"]) == pytest.approx(113.6363636363636)
    assert float(shifts.iloc[0]["pool_delta_per_10k"]) == pytest.approx(-75.75757575757575)

    # Verify full per-token diagnostics so the script cannot pass with display-only output.
    full = pd.read_csv(outdir / "lexical_full_shift_table.tsv", sep="\t")
    by_token = {str(row["token"]): row for _, row in full.iterrows()}
    assert set(by_token) == {"bonjour", "haha", "merci", "oui"}
    assert int(by_token["bonjour"]["total_count"]) == 1
    assert int(by_token["merci"]["source_count"]) == 2
    assert float(by_token["haha"]["pooled_per_10k"]) == pytest.approx(2500.0)
    assert float(by_token["oui"]["delta_cap_a1_minus_pooled_per_10k"]) == pytest.approx(-75.75757575757575)

    # Source-size summaries should reflect the post-tokenization toy source masses.
    assert _summary_lookup(outdir / "lexical_source_size_summary.tsv") == {
        "sources": "3",
        "tokens": "8",
        "top 10% token share": "37.5% (1 sources)",
        "bottom 50% token share": "25.0% (1 sources)",
        "median tokens/source": "3",
        "5th percentile tokens/source": "2",
        "max tokens/source": "3",
    }

    # Confirm Markdown and LaTeX outputs include the generated report content.
    markdown = (outdir / "lexical_tables.md").read_text(encoding="utf-8")
    assert "| POOL" in markdown
    latex = (outdir / "lexical_tables.tex").read_text(encoding="utf-8")
    assert "Total variation distances" in latex
    assert "Largest signed lexical shifts" in latex
    assert "\\label{tab:lexical-tv}" in latex


def test_reproduction_clis_match_build_outputs(tmp_path: Path) -> None:
    """The public CLI paths should match the importable build functions."""
    emoji_api = tmp_path / "emoji_api"
    emoji_cli = tmp_path / "emoji_cli"
    lexical_api = tmp_path / "lexical_api"
    lexical_cli = tmp_path / "lexical_cli"

    # Generate API reference outputs for both paper-facing real-data scripts.
    build_emoji_outputs(
        input_path="data/toy/toy_emoji.tsv",
        outdir=str(emoji_api),
        source_col="source_id",
        sequence_col="sequence",
        sep="\t",
        delimiter="-",
        min_freq=1,
        k=2,
    )
    build_lexical_outputs(
        input_path="data/toy/toy_lexical.tsv",
        outdir=str(lexical_api),
        source_col="source_id",
        text_col="text",
        sep="\t",
        min_freq=1,
        token_regex=r"(?u)\b\w+\b",
        exclude_source_regex=r"(?i)^(#?wns\.system|#?system|system)$",
        contrast="cap_a1",
        k=2,
    )

    # Run the public scripts through their command-line interfaces.
    emoji_run = _run_script(
        "reproduce_emoji_table.py",
        "--input", "data/toy/toy_emoji.tsv",
        "--outdir", str(emoji_cli),
        "--k", "2",
    )
    lexical_run = _run_script(
        "reproduce_lexical_tables.py",
        "--input", "data/toy/toy_lexical.tsv",
        "--outdir", str(lexical_cli),
        "--k", "2",
    )

    # The CLI should report the same key diagnostics and write identical TSVs.
    assert "Loaded emoji dataset: S=3 sources, V=3 types, total tokens=9" in emoji_run.stdout
    assert "Filtered lexical dataset: S=3 sources, V=4 types, total tokens=8" in lexical_run.stdout
    for filename in ["emoji_topk_table.tsv", "emoji_source_size_summary.tsv"]:
        assert (emoji_cli / filename).read_text(encoding="utf-8") == (emoji_api / filename).read_text(encoding="utf-8")
    for filename in ["lexical_tv_table.tsv", "lexical_shift_table.tsv", "lexical_full_shift_table.tsv"]:
        assert (lexical_cli / filename).read_text(encoding="utf-8") == (lexical_api / filename).read_text(encoding="utf-8")


def test_reproduce_simulation_cli_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    """Repeated CLI runs with the same seed schedule should be byte-identical."""
    first = tmp_path / "sim_first"
    second = tmp_path / "sim_second"

    # Run the public simulation script twice with the same explicit seed schedule.
    _run_script(
        "reproduce_simulation.py",
        "--quick",
        "--no-consolidate",
        "--seed-start", "7",
        "--n-seeds", "2",
        "--outdir", str(first),
    )
    _run_script(
        "reproduce_simulation.py",
        "--quick",
        "--no-consolidate",
        "--seed-start", "7",
        "--n-seeds", "2",
        "--outdir", str(second),
    )

    # The exact public artifacts should match, not only their row counts.
    assert (first / "eval" / "results.tsv").read_bytes() == (second / "eval" / "results.tsv").read_bytes()
    assert (first / "eval" / "grid.json").read_bytes() == (second / "eval" / "grid.json").read_bytes()


def test_reproduce_simulation_cli_wires_seed_schedule(tmp_path: Path) -> None:
    """Changing the seed schedule should alter metadata and simulated values."""
    seed0 = tmp_path / "sim_seed0"
    seed1 = tmp_path / "sim_seed1"

    # Run one-seed quick simulations with different top-level seed starts.
    _run_script(
        "reproduce_simulation.py",
        "--quick",
        "--no-consolidate",
        "--seed-start", "0",
        "--n-seeds", "1",
        "--outdir", str(seed0),
    )
    _run_script(
        "reproduce_simulation.py",
        "--quick",
        "--no-consolidate",
        "--seed-start", "1",
        "--n-seeds", "1",
        "--outdir", str(seed1),
    )

    # Metadata should expose the seed schedule so releases are auditable.
    grid0 = json.loads((seed0 / "eval" / "grid.json").read_text(encoding="utf-8"))
    grid1 = json.loads((seed1 / "eval" / "grid.json").read_text(encoding="utf-8"))
    assert grid0["seed_start"] == 0
    assert grid1["seed_start"] == 1
    assert grid0["seed_schedule"] == "range(seed_start, seed_start + n_seeds)"
    assert grid1["seed_schedule"] == "range(seed_start, seed_start + n_seeds)"

    # The seed column and at least some TV values should change across runs.
    results0 = pd.read_csv(seed0 / "eval" / "results.tsv", sep="\t")
    results1 = pd.read_csv(seed1 / "eval" / "results.tsv", sep="\t")
    assert set(results0["seed"]) == {0}
    assert set(results1["seed"]) == {1}
    assert not np.allclose(results0["tv_qhat_qstar"].to_numpy(), results1["tv_qhat_qstar"].to_numpy())


def test_reproduce_simulation_cli_writes_documented_output_contract(tmp_path: Path) -> None:
    """Simulation CLI should write the documented files and key columns."""
    outdir = tmp_path / "simulation"

    # Let the public command consolidate so the full output contract is exercised.
    _run_script(
        "reproduce_simulation.py",
        "--quick",
        "--seed-start", "3",
        "--n-seeds", "1",
        "--outdir", str(outdir),
    )

    expected_files = [
        outdir / "eval" / "results.tsv",
        outdir / "eval" / "grid.json",
        outdir / "consolidated" / "tables" / "coupling_grid_medians.tsv",
        outdir / "consolidated" / "figures" / "fig1_main_results.pdf",
        outdir / "consolidated" / "figures" / "fig2_alpha_sweep.pdf",
    ]
    for path in expected_files:
        assert path.exists(), path
        assert path.stat().st_size > 0, path

    # Results keep the archived evaluation schema needed by consolidation.
    results = pd.read_csv(outdir / "eval" / "results.tsv", sep="\t")
    assert list(results.columns) == [
        "status",
        "error_message",
        "generator_id",
        "sample_size",
        "vocab_size",
        "token_shape",
        "n_sources",
        "n_regimes_truth",
        "imbalance_mode",
        "imbalance_level",
        "seed",
        "method_id",
        "tv_qhat_qstar",
    ]
    assert len(results) == 3 * 4 * 1 * 7

    # The consolidated summary exposes the dimensions plotted in the paper figures.
    summary = pd.read_csv(outdir / "consolidated" / "tables" / "coupling_grid_medians.tsv", sep="\t")
    assert list(summary.columns) == ["imb_idx", "imbalance", "coupling", "method_id", "median", "q25", "q75", "count"]
    assert set(summary["method_id"]) == {
        "pooled_mle",
        "unif_sources",
        "capped_mass:alpha=0.25",
        "capped_mass:alpha=0.5",
        "capped_mass:alpha=1.0",
        "capped_mass:alpha=2",
        "capped_mass:alpha=4",
    }

    # Grid metadata should describe the seed policy and method list explicitly.
    grid = json.loads((outdir / "eval" / "grid.json").read_text(encoding="utf-8"))
    assert grid["seed_start"] == 3
    assert grid["n_seeds"] == 1
    assert grid["methods"] == [
        "pooled_mle",
        "unif_sources",
        "capped_mass:alpha=0.25",
        "capped_mass:alpha=0.5",
        "capped_mass:alpha=1.0",
        "capped_mass:alpha=2",
        "capped_mass:alpha=4",
    ]

