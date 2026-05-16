# Data access

## Public data in this repository

This repository may include small toy inputs for smoke tests and examples. These toy files are synthetic and are not derived from the *What's New, Switzerland?* corpus.

## WNS corpus data

The real-data analyses in the JADT 2026 paper use inputs derived from the *What's New, Switzerland?* corpus. The corpus is a controlled-access research corpus available to researchers via SwissUBase under a license agreement.

This repository does **not** redistribute WNS-derived token-level data, including:

- the emoji TSV used for the emoji estimator comparison;
- the lexical TSV used for the lexical sensitivity check;
- any original WhatsApp message text;
- any row-level WNS-derived token sequence data.

## Reproducing the WNS analyses

Authorized WNS users will be able to reproduce the real-data analyses as follows:

1. Obtain access to the WNS corpus through SwissUBase under the applicable license agreement.
2. Place a local copy of the corpus in the input directory documented by the preparation scripts.
3. Run the WNS preparation scripts to generate the paper-specific TSV inputs locally.
4. Run the estimator and table-generation scripts on those local TSV files.

The preparation scripts are intended to generate two paper inputs:

- an emoji-sequence TSV with columns such as `source_id` and `sequence`;
- a lexical TSV with columns such as `source_id` and `text`.

The exact column names used by scripts will be documented in the final reproduction instructions.

## Derived aggregate outputs

The repository may include aggregate tables and figures reported in the paper, provided they do not redistribute token-level WNS-derived data beyond what appears in the published article.
