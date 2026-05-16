# Reproduction plan

This document records the intended reproduction workflow for the JADT 2026 paper.

## 1. Simulations

The public reproduction package will include all code required to reproduce the simulation study:

1. generate synthetic source-by-token count matrices under the paper's imbalance/coupling regimes;
2. compute `POOL`, `UNIF`, and `CAP` estimators;
3. compute total-variation distances to the known synthetic target;
4. regenerate the simulation figure(s) and table(s).

No controlled-access corpus data are required for the simulation workflow.

## 2. Emoji real-data analysis

The emoji analysis is based on a WNS-derived emoji TSV generated from an authorized local copy of the corpus. The final preparation script should:

1. extract emoji sequences from the WNS corpus;
2. remove variation selectors and skin-tone modifiers;
3. retain gendered emoji sequences;
4. write a local TSV suitable for the estimator scripts.

The table-generation script should reproduce the paper's top-k emoji estimator table and associated total-variation summaries.

## 3. Lexical sensitivity check

The lexical check is based on message text from an authorized local copy of the WNS corpus. The final preparation/table scripts should:

1. exclude non-human/system pseudo-sources;
2. lowercase text and tokenize with the documented word-token rule;
3. remove WNS de-identification/media/notification/system placeholders and simple technical artifacts;
4. compute `POOL`, `UNIF`, and representative `CAP` estimates;
5. reproduce the final lexical TV summary and shifted-token table.

## 4. Expected command pattern

The final package should support commands following this pattern:

```bash
# Simulations
python scripts/reproduce_simulation.py --outdir outputs/simulation

# WNS conversion, for authorized users only
python scripts/prepare_wns_emoji_tsv.py --wns-root _data_private/wns --out data/private/wns_emoji.tsv
python scripts/prepare_wns_lexical_tsv.py --wns-root _data_private/wns --out data/private/wns_lexical.tsv

# Real-data tables
python scripts/reproduce_emoji_table.py --dataset data/private/wns_emoji.tsv --outdir outputs/emoji
python scripts/reproduce_lexical_table.py --dataset data/private/wns_lexical.tsv --outdir outputs/lexical
```

The exact command names may change as the reproduction package is finalized.
