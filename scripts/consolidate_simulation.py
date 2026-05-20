#!/usr/bin/env python
"""Consolidate seeded simulation results into JADT tables and figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow direct script execution from an editable checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from typical_source_estimation.simulation import (
    DEFAULT_CONFIG,
    OKABE_COLORS,
    SimulationConfig,
    decode_conditions,
    format_alpha,
    summarize_simulation,
)


RC = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.markersize": 5,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


ESTIMATORS = {
    "pooled_mle": ("pool", OKABE_COLORS["pool"], "s", 1.8),
    "unif_sources": ("unif", OKABE_COLORS["unif"], "^", 1.8),
    "capped_mass:alpha=1.0": ("cap ($\\alpha=1$)", OKABE_COLORS["cap"], "o", 1.8),
}


def _load_results(outdir: Path) -> pd.DataFrame:
    """Load successful seeded simulation rows from ``outdir``."""
    path = outdir / "eval" / "results.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Simulation results not found: {path}")

    # Keep only successful rows, matching the archived consolidation contract.
    frame = pd.read_csv(path, sep="\t")
    ok = frame[frame["status"] == "ok"].copy()
    print(f"Loaded simulation rows: {len(ok)} / {len(frame)} ok")
    return ok


def make_grid_figure(summary: pd.DataFrame, outpath: Path, *, config: SimulationConfig) -> None:
    """Write the main 1x3 coupling/imbalance simulation figure."""
    width_in = 16 / 2.54
    height_in = width_in * 0.38

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(config.imbalance_levels), figsize=(width_in, height_in), sharey=True)
        axes = np.atleast_1d(axes)

        # Set a common y-limit from the displayed estimator interquartile ranges.
        displayed = summary[summary["method_id"].isin(ESTIMATORS)]
        ymax = float(displayed["q75"].max()) * 1.12 if not displayed.empty else 0.1
        ymax = max(ymax, 0.01)

        for col, level in enumerate(config.imbalance_levels):
            ax = axes[col]
            cell = summary[summary["imb_idx"] == int(level.index)]

            # Draw estimator medians and IQR bands across coupling levels.
            for method_id, (label, color, marker, line_width) in ESTIMATORS.items():
                method_data = cell[cell["method_id"] == method_id].sort_values("coupling")
                if method_data.empty:
                    continue
                ax.plot(
                    method_data["coupling"],
                    method_data["median"],
                    color=color,
                    linestyle="-",
                    linewidth=line_width,
                    marker=marker,
                    markersize=4,
                    label=label if col == 0 else "_nolegend_",
                )
                ax.fill_between(method_data["coupling"], method_data["q25"], method_data["q75"], color=color, alpha=0.15)

            ax.set_title(_imbalance_title(level.slug), fontsize=9, pad=4)
            ax.set_xticks(list(config.coupling_levels))
            ax.set_xticklabels(["0", "0.33", "0.67", "1"], fontsize=8)
            ax.set_ylim(0, ymax)

            # Label only the outer/shared axes to keep the figure compact.
            if col == 0:
                ax.set_ylabel("TV distance to $q^*$", fontsize=9)
            if col == 1:
                ax.set_xlabel("Size--content coupling ($c$)", fontsize=9)

        axes[0].legend(loc="upper left", framealpha=0.9, fontsize=8)
        fig.tight_layout()
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote: {outpath}")


def _imbalance_title(slug: str) -> str:
    """Return the paper-facing panel title for an imbalance slug."""
    if slug == "equal":
        return "No imbalance"
    if slug == "80/20":
        return "Mild imbalance (80/20)"
    if slug == "95/5":
        return "Strong imbalance (95/5)"
    return str(slug)


def make_alpha_sweep_figure(results: pd.DataFrame, outpath: Path, *, config: SimulationConfig) -> None:
    """Write the CAP alpha-sweep figure for the mild-imbalance maximum-coupling cell."""
    decoded = decode_conditions(results, config=config)
    cell = decoded[(decoded["imb_idx"] == 1) & (decoded["coupling"] == 1.0)]
    if cell.empty:
        raise ValueError("no simulation data found for the 80/20, c=1.0 alpha-sweep cell")

    width_in = 16 / 2.54
    height_in = width_in * 0.45
    alpha_values = np.asarray(config.alpha_values, dtype=float)

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(width_in, height_in))
        ax.set_xscale("log", base=2)

        # Draw pool/unif reference lines with labels nudged above the lines.
        reference_labels: list[tuple[str, str, float]] = []
        for method_id, label, color in [
            ("pooled_mle", "pool", OKABE_COLORS["pool"]),
            ("unif_sources", "unif", OKABE_COLORS["unif"]),
        ]:
            method_data = cell[cell["method_id"] == method_id]
            if method_data.empty:
                continue
            median_value = float(method_data["tv_qhat_qstar"].median())
            ax.axhline(median_value, color=color, linestyle="--", linewidth=1.4, zorder=2)
            reference_labels.append((label, color, median_value))

        # Compute CAP median and IQR for each alpha value.
        medians: list[float] = []
        q25s: list[float] = []
        q75s: list[float] = []
        for alpha in alpha_values:
            method_id = f"capped_mass:alpha={format_alpha(float(alpha))}"
            method_data = cell[cell["method_id"] == method_id]
            medians.append(float(method_data["tv_qhat_qstar"].median()))
            q25s.append(float(method_data["tv_qhat_qstar"].quantile(0.25)))
            q75s.append(float(method_data["tv_qhat_qstar"].quantile(0.75)))

        # Use a true log2 alpha axis while keeping the original compact label.
        med = np.asarray(medians, dtype=float)
        q25 = np.asarray(q25s, dtype=float)
        q75 = np.asarray(q75s, dtype=float)
        ax.fill_between(alpha_values, q25, q75, color=OKABE_COLORS["cap"], alpha=0.20, zorder=1)
        ax.plot(alpha_values, med, color=OKABE_COLORS["cap"], linewidth=1.6, solid_capstyle="round", zorder=3)

        ax.set_xticks(alpha_values)
        ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"], fontsize=8)
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlim(alpha_values[0] / 1.15, alpha_values[-1] * 1.45)
        ax.set_ylabel("TV distance to $q^*$", fontsize=9)
        ax.set_xlabel("Cap parameter ($\\alpha$)", fontsize=9)

        # Set y-limits from visible data and add right-margin inline labels.
        pool_median = float(cell[cell["method_id"] == "pooled_mle"]["tv_qhat_qstar"].median())
        unif_median = float(cell[cell["method_id"] == "unif_sources"]["tv_qhat_qstar"].median())
        ax.set_ylim(min(unif_median, float(np.nanmin(q25))) * 0.88, pool_median * 1.12)
        y_min, y_max = ax.get_ylim()
        y_nudge = 0.018 * (y_max - y_min)
        for label, color, median_value in reference_labels:
            ax.text(alpha_values[-1] * 1.08, median_value + y_nudge, label, color=color, fontsize=8, va="bottom", ha="left")

        fig.subplots_adjust(right=0.82)
        fig.tight_layout(rect=[0, 0, 0.82, 1])
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote: {outpath}")


def consolidate_simulation_outputs(outdir: Path, *, config: SimulationConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Read simulation results and write paper-facing summaries and figures."""
    outdir = Path(outdir)
    consolidated = outdir / "consolidated"
    tables_dir = consolidated / "tables"
    figures_dir = consolidated / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load results, summarize TV distributions, and persist the summary table.
    results = _load_results(outdir)
    summary = summarize_simulation(results)
    table_path = tables_dir / "coupling_grid_medians.tsv"
    summary.to_csv(table_path, sep="\t", index=False, float_format="%.12g")
    print(f"Wrote: {table_path}")

    # Regenerate the two paper simulation figures from the seeded output.
    make_grid_figure(summary, figures_dir / "fig1_main_results.pdf", config=config)
    make_alpha_sweep_figure(results, figures_dir / "fig2_alpha_sweep.pdf", config=config)
    return summary


def main() -> None:
    """Parse command-line arguments and consolidate an existing simulation run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=str, default="outputs/simulation")
    args = parser.parse_args()
    consolidate_simulation_outputs(Path(args.outdir), config=DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
