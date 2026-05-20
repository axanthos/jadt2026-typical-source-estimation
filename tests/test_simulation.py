"""Tests for the seeded synthetic simulation workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from typical_source_estimation.simulation import (
    ImbalanceLevel,
    SimulationConfig,
    decode_conditions,
    generate_coupling_dataset,
    run_coupling_simulation,
    summarize_simulation,
    _allocate_top_split,
)


def tiny_config() -> SimulationConfig:
    """Return a small but complete simulation configuration for tests."""
    return SimulationConfig(
        n_sources=12,
        vocab_size=40,
        total_tokens=600,
        zipf_exponent=1.1,
        n_regimes=3,
        target_tv=0.2,
        coupling_levels=(0.0, 1.0),
        alpha_values=(0.5, 1.0, 2.0),
        imbalance_levels=(
            ImbalanceLevel(0, "Equal", "equal", 1.0, 1.0),
            ImbalanceLevel(1, "80/20", "80/20", 0.25, 0.8),
        ),
    )


def test_top_split_uses_ceiling_for_top_fraction() -> None:
    """Fractional top-source counts should round up for imbalance regimes."""
    totals, top_mask = _allocate_top_split(
        n_sources=6,
        total_tokens=60,
        top_fraction=0.25,
        top_mass_fraction=0.8,
        rng=np.random.default_rng(0),
    )

    assert int(top_mask.sum()) == 2
    assert float(totals[top_mask].sum()) == 48.0
    assert float(totals[~top_mask].sum()) == 12.0


def test_generate_coupling_dataset_is_seed_deterministic() -> None:
    """The same cell and seed should generate identical counts and target."""
    config = tiny_config()
    level = config.imbalance_levels[1]

    first = generate_coupling_dataset(config=config, level=level, coupling=1.0, seed=4)
    second = generate_coupling_dataset(config=config, level=level, coupling=1.0, seed=4)

    assert np.array_equal(first.dataset.counts, second.dataset.counts)
    assert np.allclose(first.q_star, second.q_star)
    assert first.metadata["n_regimes_truth"] == 110


def test_generate_coupling_dataset_preserves_mass_and_labels() -> None:
    """Generated datasets should preserve the requested dimensions and mass."""
    config = tiny_config()
    generated = generate_coupling_dataset(
        config=config,
        level=config.imbalance_levels[0],
        coupling=0.0,
        seed=2,
    )

    assert generated.dataset.S == 12
    assert generated.dataset.V == 40
    assert generated.dataset.total_mass == 600
    assert generated.dataset.source_ids[0] == "s0000"
    assert generated.dataset.vocab[0] == "t00000"
    assert np.isclose(generated.q_star.sum(), 1.0)


def test_run_coupling_simulation_writes_expected_schema(tmp_path: Path) -> None:
    """The runner should write one result row per condition/seed/method."""
    config = tiny_config()
    results = run_coupling_simulation(outdir=tmp_path, config=config, seed_start=10, n_seeds=2)

    expected_rows = 2 * 2 * 2 * 5
    assert len(results) == expected_rows
    assert (tmp_path / "eval" / "results.tsv").exists()
    assert (tmp_path / "eval" / "grid.json").exists()
    assert set(results["status"]) == {"ok"}
    assert results["tv_qhat_qstar"].between(0.0, 1.0).all()


def test_summarize_simulation_decodes_conditions() -> None:
    """Summary rows should expose decoded imbalance and coupling dimensions."""
    frame = pd.DataFrame(
        [
            {"status": "ok", "n_regimes_truth": 110, "method_id": "pooled_mle", "tv_qhat_qstar": 0.2},
            {"status": "ok", "n_regimes_truth": 110, "method_id": "pooled_mle", "tv_qhat_qstar": 0.4},
            {"status": "ok", "n_regimes_truth": 110, "method_id": "unif_sources", "tv_qhat_qstar": 0.1},
        ]
    )

    decoded = decode_conditions(frame)
    summary = summarize_simulation(frame)

    assert decoded.loc[0, "imb_idx"] == 1
    assert decoded.loc[0, "coupling"] == 1.0
    assert decoded.loc[0, "imbalance"] == "80/20"
    pooled = summary[summary["method_id"] == "pooled_mle"].iloc[0]
    assert pooled["median"] == np.float64(0.3) or np.isclose(pooled["median"], 0.3)
    assert pooled["count"] == 2
