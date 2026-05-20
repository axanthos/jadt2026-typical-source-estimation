#!/usr/bin/env python
"""Run the seeded JADT simulation grid and optionally consolidate outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow direct script execution from an editable checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typical_source_estimation.simulation import DEFAULT_CONFIG, SimulationConfig, run_coupling_simulation
from consolidate_simulation import consolidate_simulation_outputs


def _quick_config() -> SimulationConfig:
    """Return a reduced configuration for fast smoke runs."""
    return SimulationConfig(
        n_sources=12,
        vocab_size=40,
        total_tokens=600,
        zipf_exponent=DEFAULT_CONFIG.zipf_exponent,
        n_regimes=DEFAULT_CONFIG.n_regimes,
        target_tv=DEFAULT_CONFIG.target_tv,
        coupling_levels=DEFAULT_CONFIG.coupling_levels,
        alpha_values=DEFAULT_CONFIG.alpha_values,
        imbalance_levels=DEFAULT_CONFIG.imbalance_levels,
    )


def main() -> None:
    """Parse command-line arguments and run the simulation workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=str, default="outputs/simulation")
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="Use a smaller dataset and at most the requested seed count.")
    parser.add_argument("--no-consolidate", action="store_true", help="Only write eval/results.tsv and grid metadata.")
    args = parser.parse_args()

    # Quick mode keeps the full condition/method grid but lowers data volume.
    config = _quick_config() if bool(args.quick) else DEFAULT_CONFIG
    n_seeds = min(int(args.n_seeds), 3) if bool(args.quick) else int(args.n_seeds)

    print(f"Running simulation: outdir={args.outdir}")
    print(f"Seed schedule: range({int(args.seed_start)}, {int(args.seed_start) + n_seeds})")
    print(
        f"Grid: imbalances={len(config.imbalance_levels)}, couplings={len(config.coupling_levels)}, "
        f"alphas={len(config.alpha_values)}, seeds={n_seeds}"
    )
    results = run_coupling_simulation(
        outdir=Path(args.outdir),
        config=config,
        seed_start=int(args.seed_start),
        n_seeds=n_seeds,
    )
    print(f"Wrote: {Path(args.outdir) / 'eval' / 'results.tsv'} ({len(results)} rows)")

    # Consolidate by default so a single command regenerates paper-facing outputs.
    if not bool(args.no_consolidate):
        consolidate_simulation_outputs(Path(args.outdir), config=config)


if __name__ == "__main__":
    main()
