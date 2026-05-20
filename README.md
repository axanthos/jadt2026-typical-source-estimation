# JADT 2026 typical-source estimation reproduction package

This repository is the paper-specific reproduction package for:

> Aris Xanthos. 2026. *Estimating the typical-source distribution in imbalanced corpora*. JADT 2026.

It contains code, scripts, documentation, and small toy inputs for reproducing
the simulation study and estimator-comparison tables/figures reported in the
paper.

## Status

Release-ready reproduction package with the source-count data model, the three
paper estimators, total-variation distance, source-size summaries, preparation
scripts for the What's New, Switzerland? (WNS) corpus, final real-data table
scripts, seeded simulation scripts, and tests.

## What this repository contains

- Implementations of the population estimators discussed in the paper:
  - `POOL`: pooled maximum-likelihood estimate;
  - `UNIF`: uniform average over sources;
  - `CAP`: capped source-mass estimator.
- Core utilities for source-by-token count datasets.
- Total-variation distance and source-size imbalance summaries.
- Small synthetic toy inputs for examples and smoke tests.
- Preparation scripts that regenerate paper-specific WNS TSV inputs locally for authorized users.
- Emoji and lexical table-generation scripts consuming prepared TSV inputs.
- Seeded simulation scripts that regenerate paper-facing simulation summaries and figures.
- Documentation for data access, simulation provenance, and the reproduction workflow.

## What this repository will not contain

This repository does **not** redistribute token-level data derived from WNS.

The dataset is available on demand for research purposes, under a restricted
license contract, from the SWISSUbase repository (https://www.swissubase.ch).
In accordance with the corpus privacy commitments, WNS-derived token-level
inputs used in the paper are not published here. Authorized WNS users can
regenerate them locally with the preparation scripts provided in this
repository.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── data/
│   ├── README.md
│   └── toy/
├── docs/
│   ├── data_access.md
│   ├── reproduction_plan.md
│   └── simulation_provenance.md
├── configs/
│   └── wns_jadt_preprocessing.ini
├── scripts/
│   ├── README.md
│   ├── prepare_wns_posts_tsv.py
│   ├── prepare_wns_emoji_tsv.py
│   ├── prepare_wns_lexical_tsv.py
│   ├── reproduce_emoji_table.py
│   ├── reproduce_lexical_tables.py
│   └── reproduce_simulation.py
├── src/
│   └── typical_source_estimation/
└── tests/
```

## Installation

The package declares its runtime dependencies in `pyproject.toml`, including
`numpy`, `pandas`, `matplotlib`, `emoji`, and `lxml`. The `dev` extra adds
`pytest` for the test suite.

With `uv`, set up the development environment and run the tests with:

```bash
uv sync --extra dev
uv run python -m pytest -q
```

With standard `pip`, use an editable install with the development extra:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Minimal example

```python
from typical_source_estimation import load_sequence_tsv, pooled_mle, uniform_sources, capped_mass_alpha

# Load a small source/sequence table.
ds = load_sequence_tsv("data/toy/toy_emoji.tsv")

# Compute the three paper estimators.
q_pool = pooled_mle(ds).q_hat
q_unif = uniform_sources(ds).q_hat
q_cap = capped_mass_alpha(ds, alpha=1.0).q_hat
```

## Data access and reproduction

The real-data analyses in the paper use two TSV inputs derived from WNS:

1. an emoji-sequence table;
2. a lexical message-text table.

For public reproducibility, this repository provides:

- conversion scripts that authorized users can run on a local copy of WNS;
- estimator/table-generation scripts that consume the derived local TSV files;
- toy inputs with the same column conventions.

For details, see:

- [Data access](docs/data_access.md)
- [Reproduction plan](docs/reproduction_plan.md)
- [Simulation provenance](docs/simulation_provenance.md)

## Citation

Please cite both the accompanying JADT paper and the archived software release.
The paper citation is, pending final proceedings metadata:

> Xanthos, Aris. 2026. *Estimating the typical-source distribution in imbalanced corpora*. JADT 2026.

When using the WNS real-data analyses, also cite the dataset as:

> Xanthos, A., Gupta, P., Benkais, L., Doudot, L., & Grütter, A. (2024). What's New, Switzerland? Corpus (Version 1.0.0) [Data set]. LaRS - Language Repository of Switzerland. https://doi.org/10.48656/pa3t-xh52

A `CITATION.cff` file is included for software citation metadata. Add the Zenodo
DOI to `CITATION.cff` after the archival release has been minted.

## License

Code in this repository is released under the MIT License unless otherwise
stated. See [LICENSE](LICENSE).

No license is granted here for WNS-derived token-level data, because such data
are not redistributed in this repository.
