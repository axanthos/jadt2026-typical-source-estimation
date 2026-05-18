# JADT 2026 typical-source estimation reproduction package

This repository is the paper-specific reproduction package for:

> Aris Xanthos. 2026. *Estimating the typical-source distribution in imbalanced corpora*. JADT 2026.

It contains code, scripts, documentation, and small toy inputs for reproducing
the simulation study and estimator-comparison tables/figures reported in the
paper.

## Status

Development scaffold with the core source-count data model, the three paper
estimators, total-variation distance, source-size summaries, unit tests, and
initial WNS preparation scripts for authorized local reproduction. Simulation
and final table-generation scripts will be added in subsequent steps.

## What this repository contains

- Implementations of the population estimators discussed in the paper:
  - `POOL`: pooled maximum-likelihood estimate;
  - `UNIF`: uniform average over sources;
  - `CAP`: capped source-mass estimator.
- Core utilities for source-by-token count datasets.
- Total-variation distance and source-size imbalance summaries.
- Small synthetic toy inputs for examples and smoke tests.
- WNS preparation scripts that regenerate paper-specific TSV inputs locally for authorized users.
- Documentation for the planned reproduction workflow.

## What this repository will not contain

This repository does **not** redistribute token-level data derived from the
*What's New, Switzerland?* corpus.

The WNS corpus is a controlled-access research corpus available to researchers
via SwissUBase under a license agreement. In accordance with the corpus privacy
commitments, WNS-derived token-level inputs used in the paper are not published
here. Authorized WNS users will be able to regenerate them locally with the
preparation scripts provided in this repository.

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
│   └── reproduction_plan.md
├── configs/
│   └── wns_jadt_preprocessing.ini
├── scripts/
│   ├── README.md
│   ├── prepare_wns_posts_tsv.py
│   ├── prepare_wns_emoji_tsv.py
│   └── prepare_wns_lexical_tsv.py
├── src/
│   └── typical_source_estimation/
└── tests/
```

## Installation

The package declares its runtime dependencies in `pyproject.toml`, including
`numpy`, `pandas`, and `matplotlib`.  The `dev` extra adds `pytest` for the test
suite.

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

The real-data analyses in the paper use two TSV inputs derived from the WNS
corpus:

1. an emoji-sequence table;
2. a lexical message-text table.

For public reproducibility, this repository will provide:

- conversion scripts that authorized users can run on a local copy of the WNS
  corpus;
- estimator/table-generation scripts that consume the derived local TSV files;
- toy inputs with the same column conventions.

For details, see [`docs/data_access.md`](docs/data_access.md) and
[`docs/reproduction_plan.md`](docs/reproduction_plan.md).

## Citation

Please cite the JADT 2026 paper and, when using the WNS real-data analyses, also
cite the WNS corpus through its official SwissUBase citation.

A `CITATION.cff` file is included for software citation metadata and will be
updated when the archival release DOI is available.

## License

Code in this repository is released under the MIT License unless otherwise
stated. See [`LICENSE`](LICENSE).

No license is granted here for WNS-derived token-level data, because such data
are not redistributed in this repository.
