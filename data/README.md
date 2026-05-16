# Data directory

This directory is reserved for public toy data and local private inputs.

## Public toy data

The `toy/` subdirectory contains small synthetic examples for smoke tests and documentation. These files are not derived from the WNS corpus.

## Private controlled-access data

Do not commit WNS-derived token-level inputs to this repository.

Authorized WNS users should generate local private TSV files using the preparation scripts and place them under an ignored directory such as:

```text
data/private/
```

The `.gitignore` file excludes `data/private/` and `_data_private/` by default.
