"""Seeded synthetic simulation workflow for the JADT reproduction package.

The paper's simulation study evaluates three estimators under controlled
source-size imbalance and size--content coupling: pooled token weighting
(``POOL``), uniform source weighting (``UNIF``), and capped source-mass
weighting (``CAP``). This module is a compact, seeded implementation of that
paper-specific workflow with an explicit seed schedule for release-grade
reproduction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import json
import math

import numpy as np
import pandas as pd

from typical_source_estimation.data import CountsDataset
from typical_source_estimation.estimators import capped_mass_alpha, pooled_mle, uniform_sources
from typical_source_estimation.metrics import total_variation


@dataclass(frozen=True, slots=True)
class ImbalanceLevel:
    """Source-size imbalance setting used by the coupling-spectrum grid."""

    index: int
    label: str
    slug: str
    top_fraction: float
    top_mass_fraction: float


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Configuration for the seeded coupling-spectrum simulation."""

    n_sources: int = 100
    vocab_size: int = 1000
    total_tokens: int = 100_000
    zipf_exponent: float = 1.1
    n_regimes: int = 3
    target_tv: float = 0.2
    coupling_levels: tuple[float, ...] = (0.0, 0.33, 0.67, 1.0)
    alpha_values: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    imbalance_levels: tuple[ImbalanceLevel, ...] = field(
        default_factory=lambda: (
            ImbalanceLevel(0, "Equal", "equal", 1.0, 1.0),
            ImbalanceLevel(1, "80/20", "80/20", 0.2, 0.8),
            ImbalanceLevel(2, "95/5", "95/5", 0.05, 0.95),
        )
    )

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the configuration."""
        return {
            "n_sources": int(self.n_sources),
            "vocab_size": int(self.vocab_size),
            "total_tokens": int(self.total_tokens),
            "zipf_exponent": float(self.zipf_exponent),
            "n_regimes": int(self.n_regimes),
            "target_tv": float(self.target_tv),
            "coupling_levels": [float(v) for v in self.coupling_levels],
            "alpha_values": [float(v) for v in self.alpha_values],
            "imbalance_levels": [asdict(level) for level in self.imbalance_levels],
        }


@dataclass(frozen=True, slots=True)
class GeneratedSimulationDataset:
    """Generated count table and known typical-source target distribution."""

    dataset: CountsDataset
    q_star: np.ndarray
    metadata: dict[str, object]


# Shared display/order constants used by scripts and tests.
DEFAULT_CONFIG = SimulationConfig()
OKABE_COLORS = {"pool": "#0072B2", "unif": "#E69F00", "cap": "#009E73"}


def _zipf_distribution(vocab_size: int, exponent: float) -> np.ndarray:
    """Return a normalized Zipf-like distribution over ``vocab_size`` tokens."""
    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than 1.")
    if not np.isfinite(exponent) or float(exponent) <= 0:
        raise ValueError("exponent must be a positive finite number.")

    # Build a deterministic long-tailed base distribution over token ranks.
    ranks = np.arange(1, int(vocab_size) + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, float(exponent))
    return weights / float(weights.sum())


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    """Return total variation distance for already aligned vectors."""
    return 0.5 * float(np.abs(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)).sum())


def _allocate_top_split(
    *,
    n_sources: int,
    total_tokens: int,
    top_fraction: float,
    top_mass_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate token totals with a randomized top/bottom source split."""
    if n_sources <= 1:
        raise ValueError("n_sources must be greater than 1 for imbalanced allocation.")
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive.")
    if not (0.0 < top_fraction < 1.0):
        raise ValueError("top_fraction must be in (0, 1).")
    if not (0.0 < top_mass_fraction < 1.0):
        raise ValueError("top_mass_fraction must be in (0, 1).")

    # Select top sources by permutation so size/content coupling depends on seed.
    n_top = max(1, int(math.ceil(float(top_fraction) * int(n_sources))))
    n_top = min(n_top, int(n_sources) - 1)
    permutation = rng.permutation(int(n_sources))
    top_indices = permutation[:n_top]
    bottom_indices = permutation[n_top:]

    # Split total token mass evenly within each bucket, conserving the total.
    top_mass = int(round(float(top_mass_fraction) * int(total_tokens)))
    top_mass = min(max(top_mass, 1), int(total_tokens) - 1)
    bottom_mass = int(total_tokens) - top_mass
    totals = np.zeros(int(n_sources), dtype=np.int64)

    # Assign top bucket masses and deterministic remainders in selected order.
    top_base, top_remainder = divmod(top_mass, n_top)
    totals[top_indices] = top_base
    if top_remainder:
        totals[top_indices[:top_remainder]] += 1

    # Assign bottom bucket masses and deterministic remainders in selected order.
    n_bottom = int(n_sources) - n_top
    bottom_base, bottom_remainder = divmod(bottom_mass, n_bottom)
    totals[bottom_indices] = bottom_base
    if bottom_remainder:
        totals[bottom_indices[:bottom_remainder]] += 1

    # Return both source totals and the top-source mask used for coupling.
    top_mask = np.zeros(int(n_sources), dtype=bool)
    top_mask[top_indices] = True
    return totals.astype(np.float64), top_mask


def _allocate_equal_totals(*, n_sources: int, total_tokens: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Allocate nearly equal integer token totals across sources."""
    if n_sources <= 0:
        raise ValueError("n_sources must be positive.")
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive.")

    # Equal allocation mirrors the no-imbalance condition from the paper grid.
    base, remainder = divmod(int(total_tokens), int(n_sources))
    totals = np.full(int(n_sources), float(base), dtype=np.float64)
    if remainder:
        totals[rng.permutation(int(n_sources))[:remainder]] += 1.0
    return totals, np.zeros(int(n_sources), dtype=bool)


def _make_cluster_centroids_with_target_tv(
    *,
    base: np.ndarray,
    n_regimes: int,
    target_tv: float,
    rng: np.random.Generator,
    subset_fraction: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Create regime centroids whose mean pairwise TV is near ``target_tv``."""
    if n_regimes < 2:
        raise ValueError("n_regimes must be at least 2.")
    if not (0.0 < target_tv < 1.0):
        raise ValueError("target_tv must be in (0, 1).")

    # Choose regime-specific token subsets to receive extra mass.
    base = np.asarray(base, dtype=np.float64)
    vocab_size = int(base.size)
    permutation = rng.permutation(vocab_size)
    subset_size = max(1, int(round(float(subset_fraction) * vocab_size)))
    subsets: list[np.ndarray] = []
    for regime_idx in range(int(n_regimes)):
        start = (regime_idx * subset_size) % vocab_size
        subset = permutation[start : start + subset_size]
        if subset.size < subset_size:
            subset = np.concatenate([subset, permutation[: subset_size - subset.size]])
        subsets.append(subset)

    def build(delta: float) -> np.ndarray:
        centroids = np.zeros((int(n_regimes), vocab_size), dtype=np.float64)
        for regime_idx, subset in enumerate(subsets):
            weights = base.copy()
            weights[subset] *= 1.0 + float(delta)
            centroids[regime_idx, :] = weights / float(weights.sum())
        return centroids

    def mean_pairwise_tv(centroids: np.ndarray) -> float:
        distances = [
            _tv(centroids[i], centroids[j])
            for i in range(int(n_regimes))
            for j in range(i + 1, int(n_regimes))
        ]
        return float(np.mean(distances)) if distances else 0.0

    # Binary-search the boost parameter to match the desired separation.
    lo, hi = 0.0, 100.0
    best_centroids = build(0.0)
    best_tv = mean_pairwise_tv(best_centroids)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        centroids = build(mid)
        achieved = mean_pairwise_tv(centroids)
        best_centroids, best_tv = centroids, achieved
        if achieved < target_tv:
            lo = mid
        else:
            hi = mid
    return best_centroids, float(best_tv)


def _make_labels(prefix: str, count: int, width: int) -> list[str]:
    """Return deterministic labels such as ``s0001`` or ``t00001``."""
    return [f"{prefix}{idx:0{width}d}" for idx in range(int(count))]


def encode_condition(level: ImbalanceLevel, coupling: float) -> int:
    """Encode imbalance/coupling condition in the stable integer convention."""
    return int(level.index) * 100 + int(round(float(coupling) * 10.0))


def generate_coupling_dataset(
    *,
    config: SimulationConfig,
    level: ImbalanceLevel,
    coupling: float,
    seed: int,
) -> GeneratedSimulationDataset:
    """Generate one seeded source-count dataset for a coupling-grid cell."""
    rng = np.random.default_rng(int(seed))
    n_sources = int(config.n_sources)
    vocab_size = int(config.vocab_size)
    total_tokens = int(config.total_tokens)

    # Build the base lexical distribution and per-source token budgets.
    base = _zipf_distribution(vocab_size, float(config.zipf_exponent))
    if level.slug == "equal":
        totals, is_top = _allocate_equal_totals(n_sources=n_sources, total_tokens=total_tokens, rng=rng)
    else:
        totals, is_top = _allocate_top_split(
            n_sources=n_sources,
            total_tokens=total_tokens,
            top_fraction=float(level.top_fraction),
            top_mass_fraction=float(level.top_mass_fraction),
            rng=rng,
        )

    # Create content regimes and assign source distributions.
    centroids, achieved_tv = _make_cluster_centroids_with_target_tv(
        base=base,
        n_regimes=int(config.n_regimes),
        target_tv=float(config.target_tv),
        rng=rng,
    )
    assignments = np.zeros(n_sources, dtype=np.int64)

    # Equal-mass sources have no size-content coupling; assign regimes uniformly.
    if level.slug == "equal":
        assignments = rng.integers(low=0, high=int(config.n_regimes), size=n_sources, dtype=np.int64)
    else:
        if not (0.0 <= float(coupling) <= 1.0):
            raise ValueError("coupling must be in [0, 1].")
        top_indices = np.flatnonzero(is_top)
        bottom_indices = np.flatnonzero(~is_top)

        # Force a seed-determined fraction of large sources into regime 0.
        n_forced = int(round(float(coupling) * int(top_indices.size)))
        top_permutation = rng.permutation(int(top_indices.size)) if top_indices.size else np.array([], dtype=np.int64)
        forced = top_indices[top_permutation[:n_forced]] if top_indices.size else np.array([], dtype=np.int64)
        free_top = top_indices[top_permutation[n_forced:]] if top_indices.size else np.array([], dtype=np.int64)
        assignments[forced] = 0

        # Assign non-forced large sources and all small sources uniformly over regimes.
        if free_top.size:
            assignments[free_top] = rng.integers(low=0, high=int(config.n_regimes), size=int(free_top.size), dtype=np.int64)
        if bottom_indices.size:
            assignments[bottom_indices] = rng.integers(low=0, high=int(config.n_regimes), size=int(bottom_indices.size), dtype=np.int64)

    # Sample observed counts from the assigned source distributions.
    source_distributions = centroids[assignments]
    counts = np.zeros((n_sources, vocab_size), dtype=np.float64)
    for source_idx, source_total in enumerate(totals):
        token_count = int(round(float(source_total)))
        if token_count > 0:
            counts[source_idx, :] = rng.multinomial(token_count, source_distributions[source_idx]).astype(np.float64)

    # The known target is the typical-source average of latent source profiles.
    q_star = np.mean(source_distributions, axis=0)
    dataset = CountsDataset(
        counts=counts,
        source_ids=_make_labels("s", n_sources, 4),
        vocab=_make_labels("t", vocab_size, 5),
        metadata={"simulation": "coupling-spectrum", "seed": int(seed)},
    )
    return GeneratedSimulationDataset(
        dataset=dataset,
        q_star=q_star.astype(np.float64),
        metadata={
            "imbalance": level.label,
            "imb_idx": int(level.index),
            "coupling": float(coupling),
            "n_regimes_truth": encode_condition(level, coupling),
            "achieved_mean_pairwise_tv": float(achieved_tv),
        },
    )


def method_ids(config: SimulationConfig) -> list[str]:
    """Return the paper estimator identifiers in stable display/evaluation order."""
    ids = ["pooled_mle", "unif_sources"]
    ids.extend(f"capped_mass:alpha={format_alpha(alpha)}" for alpha in config.alpha_values)
    return ids


def format_alpha(alpha: float) -> str:
    """Format cap-alpha values using the stable method-id convention."""
    return "1.0" if abs(float(alpha) - 1.0) < 1e-12 else "%g" % float(alpha)


def estimate_by_method(dataset: CountsDataset, method_id: str) -> np.ndarray:
    """Run one paper estimator and return its distribution estimate."""
    if method_id == "pooled_mle":
        return pooled_mle(dataset).q_hat
    if method_id == "unif_sources":
        return uniform_sources(dataset).q_hat
    if method_id.startswith("capped_mass:alpha="):
        alpha = float(method_id.split("=", 1)[1])
        return capped_mass_alpha(dataset, alpha=alpha).q_hat
    raise ValueError(f"unsupported simulation method_id: {method_id!r}")


def iter_simulation_rows(
    *,
    config: SimulationConfig,
    seeds: Sequence[int],
) -> Iterable[dict[str, object]]:
    """Yield result rows for the full coupling-spectrum evaluation grid."""
    methods = method_ids(config)
    for level in config.imbalance_levels:
        for coupling in config.coupling_levels:
            for seed in [int(s) for s in seeds]:
                generated = generate_coupling_dataset(config=config, level=level, coupling=float(coupling), seed=seed)

                # Evaluate all paper estimators on the same generated dataset.
                for method_id in methods:
                    try:
                        q_hat = estimate_by_method(generated.dataset, method_id)
                        tv = total_variation(q_hat, generated.q_star)
                        status = "ok"
                        error = ""
                    except Exception as exc:  # pragma: no cover - defensive run logging.
                        tv = np.nan
                        status = "error"
                        error = repr(exc)

                    # Keep the result schema stable for consolidation and downstream checks.
                    yield {
                        "status": status,
                        "error_message": error,
                        "generator_id": "S2" if level.slug == "equal" else "S4",
                        "sample_size": int(config.total_tokens),
                        "vocab_size": int(config.vocab_size),
                        "token_shape": float(config.zipf_exponent),
                        "n_sources": int(config.n_sources),
                        "n_regimes_truth": int(generated.metadata["n_regimes_truth"]),
                        "imbalance_mode": "none",
                        "imbalance_level": 0.0,
                        "seed": int(seed),
                        "method_id": method_id,
                        "tv_qhat_qstar": float(tv),
                    }


def run_coupling_simulation(
    *,
    outdir: str | Path,
    config: SimulationConfig = DEFAULT_CONFIG,
    seed_start: int = 0,
    n_seeds: int = 100,
) -> pd.DataFrame:
    """Run the seeded simulation grid and write ``eval/results.tsv``."""
    if n_seeds <= 0:
        raise ValueError("n_seeds must be positive.")
    outdir = Path(outdir)
    eval_dir = outdir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate a deterministic sequence of integer seeds and persist metadata.
    seeds = list(range(int(seed_start), int(seed_start) + int(n_seeds)))
    rows = list(iter_simulation_rows(config=config, seeds=seeds))
    results = pd.DataFrame(rows)
    results.to_csv(eval_dir / "results.tsv", sep="\t", index=False, float_format="%.12g")

    # Record enough metadata for later provenance checks and paper updates.
    grid = {
        "config": config.to_json_dict(),
        "seed_start": int(seed_start),
        "n_seeds": int(n_seeds),
        "seed_schedule": "range(seed_start, seed_start + n_seeds)",
        "methods": method_ids(config),
    }
    (eval_dir / "grid.json").write_text(json.dumps(grid, indent=2, sort_keys=True), encoding="utf-8")
    return results


def decode_conditions(frame: pd.DataFrame, *, config: SimulationConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Decode integer condition codes into imbalance/coupling columns."""
    decoded = frame.copy()
    decoded["imb_idx"] = decoded["n_regimes_truth"].astype(int) // 100
    decoded["c_raw"] = decoded["n_regimes_truth"].astype(int) % 100

    # Map rounded integer coupling codes back to nominal paper values.
    coupling_map = {int(round(c * 10.0)): float(c) for c in config.coupling_levels}
    label_map = {int(level.index): level.label for level in config.imbalance_levels}
    decoded["coupling"] = decoded["c_raw"].map(coupling_map)
    decoded["imbalance"] = decoded["imb_idx"].map(label_map)
    return decoded


def summarize_simulation(results: pd.DataFrame) -> pd.DataFrame:
    """Compute median and interquartile summaries by condition and method."""
    ok = results[results["status"] == "ok"].copy()
    decoded = decode_conditions(ok)
    group_cols = ["imb_idx", "imbalance", "coupling", "method_id"]

    # Aggregate the seeded TV distribution used by the paper figures.
    return (
        decoded.groupby(group_cols, dropna=False)["tv_qhat_qstar"]
        .agg(
            median="median",
            q25=lambda series: series.quantile(0.25),
            q75=lambda series: series.quantile(0.75),
            count="count",
        )
        .reset_index()
    )
