# Simulation provenance

This document records the simulation design and seed schedule used for the JADT
2026 reproduction package.

## Simulation design

The simulation workflow uses a coupling-spectrum design with:

- `S = 100` sources;
- `V = 1000` token types;
- `N = 100000` tokens;
- Zipf exponent `1.1`;
- three latent regimes with target mean pairwise TV around `0.2`;
- imbalance levels `equal`, `80/20`, and `95/5`;
- coupling levels `0`, `0.33`, `0.67`, and `1`;
- CAP alpha values `0.25`, `0.5`, `1`, `2`, and `4`.

The released simulation workflow makes the random seed schedule explicit so
that all reported simulation summaries and figures can be regenerated.

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
