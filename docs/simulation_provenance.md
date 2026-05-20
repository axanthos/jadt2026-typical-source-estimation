# Simulation provenance

This document records the simulation path used for the JADT 2026 reproduction
package.

## Historical reference

The archived project generated the paper simulation outputs with a coupling-
spectrum script using:

- `S = 100` sources;
- `V = 1000` token types;
- `N = 100000` tokens;
- Zipf exponent `1.1`;
- three latent regimes with target mean pairwise TV around `0.2`;
- imbalance levels `equal`, `80/20`, and `95/5`;
- coupling levels `0`, `0.33`, `0.67`, and `1`;
- CAP alpha values `0.25`, `0.5`, `1`, `2`, and `4`.

The reproduction package ports this deterministic design but makes the seed
schedule explicit.  The historical output is treated as a reference for parity;
the seeded reproduction output is the source of truth for the release and for
camera-ready paper values once parity is checked.

## Released seed schedule

The public simulation command uses:

```bash
python scripts/reproduce_simulation.py --outdir outputs/simulation --seed-start 0 --n-seeds 100
```

This corresponds to the seed sequence:

```text
0, 1, 2, ..., 99
```

The generated `outputs/simulation/eval/grid.json` records the configuration and
seed schedule used for each run.

## Paper-facing outputs

The simulation command writes:

```text
outputs/simulation/eval/results.tsv
outputs/simulation/eval/grid.json
outputs/simulation/consolidated/tables/coupling_grid_medians.tsv
outputs/simulation/consolidated/figures/fig1_main_results.pdf
outputs/simulation/consolidated/figures/fig2_alpha_sweep.pdf
```

Figure 2 uses a true log-scaled alpha axis while keeping the axis label compact;
the caption should state that consecutive alpha ticks correspond to doubling the
cap.

## Parity check against archived summary

On the current snapshot, the seeded reproduction with `--seed-start 0 --n-seeds
100` matches the archived `out/eval_coupling3/consolidated/tables/coupling_grid_medians.tsv`
summary for the paper estimators (`POOL`, `UNIF`, and all reported `CAP` alpha
values) to numerical roundoff.  The largest median difference observed in the
comparison was below `1e-12`.
