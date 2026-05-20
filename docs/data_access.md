# Data access

## Public data in this repository

This repository includes small toy inputs for smoke tests and examples. These
files are synthetic and are not derived from the What's New, Switzerland? (WNS)
corpus.

## WNS data

The real-data analyses in the JADT 2026 paper use inputs derived from WNS.
WNS is controlled-access and available to researchers via SwissUBase under a
license agreement.

This repository does **not** redistribute WNS-derived token-level data,
including:

- the emoji TSV used for the emoji estimator comparison;
- the lexical TSV used for the lexical sensitivity check;
- any original WhatsApp message text;
- any row-level WNS-derived token sequence data.

## Reproducing the WNS analyses

Authorized WNS users will be able to reproduce the real-data analyses as
follows:

1. obtain access to WNS through SwissUBase under the applicable
   license agreement;
2. place a local copy of WNS in the private input directory documented by
   the preparation scripts;
3. run the WNS preparation scripts to generate the paper-specific TSV inputs
   locally;
4. run the estimator and table-generation scripts on those local TSV files.

The preparation scripts are intended to generate two paper inputs:

- an emoji-sequence TSV with columns `source_id` and `sequence`;
- a lexical TSV with columns `source_id` and `text`.

## Derived aggregate outputs

The repository may include aggregate tables and figures reported in the paper,
provided they do not redistribute token-level WNS-derived data beyond what
appears in the published article.
