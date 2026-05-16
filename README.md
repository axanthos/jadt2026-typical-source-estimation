# JADT 2026 typical-source estimation reproduction package

This repository is the paper-specific reproduction package for:

> Aris Xanthos. 2026. *Estimating the typical-source distribution in imbalanced corpora*. JADT 2026.

It contains code, scripts, documentation, and small toy inputs for reproducing the simulation study and the reported estimator-comparison tables/figures from the paper.

## Status

Initial scaffold. The repository is intended to be completed for the JADT 2026 camera-ready/proceedings release.

## What this repository will contain

- Implementations of the population estimators discussed in the paper:
  - `POOL`: pooled maximum-likelihood estimate;
  - `UNIF`: uniform average over sources;
  - `CAP`: capped source-mass estimator.
- Synthetic data generators and simulation scripts.
- Scripts to reproduce paper tables and figures.
- Scripts allowing authorized users of the *What's New, Switzerland?* corpus to regenerate the emoji and lexical TSV inputs used in the real-data analyses.
- Small toy datasets for smoke tests and examples.

## What this repository will **not** contain

This repository does **not** redistribute token-level data derived from the *What's New, Switzerland?* corpus.

The WNS corpus is a controlled-access research corpus available to researchers via SwissUBase under a license agreement. In accordance with the corpus privacy commitments, WNS-derived token-level inputs used in the paper are not published here. Authorized WNS users can regenerate them locally with the preparation scripts that will be provided in this repository.

## Data access and reproduction

The real-data analyses in the paper use two TSV inputs derived from the WNS corpus:

1. an emoji-sequence table;
2. a lexical message-text table.

For public reproducibility, this repository will provide:

- the conversion scripts needed to generate those TSVs from an authorized local copy of the WNS corpus;
- the estimator/table-generation scripts that consume those TSVs;
- toy inputs with the same column conventions.

For details, see [`docs/data_access.md`](docs/data_access.md) and [`docs/reproduction_plan.md`](docs/reproduction_plan.md).

## Repository layout

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── docs/
│   ├── data_access.md
│   └── reproduction_plan.md
├── scripts/
│   └── README.md
├── src/
│   └── typical_source_estimation/
├── tests/
└── data/
    ├── README.md
    └── toy/
```

## Installation

The final package is expected to support a standard Python workflow. During development:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

## Citation

Please cite the JADT 2026 paper and, when using the WNS real-data analyses, also cite the WNS corpus through its official SwissUBase citation.

A `CITATION.cff` file is included for software citation metadata and will be updated when the archival release DOI is available.

## License

Code in this repository is released under the MIT License unless otherwise stated. See [`LICENSE`](LICENSE).

No license is granted here for WNS-derived token-level data, because such data are not redistributed in this repository.
