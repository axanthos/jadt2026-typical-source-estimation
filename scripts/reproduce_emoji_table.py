"""Reproduce the JADT emoji Top-K and source-size tables.

The script consumes the two-column emoji TSV generated locally from the
controlled-access WNS corpus.  It does not require the corpus itself once that
TSV exists, and it writes the TSV, Markdown, and LaTeX outputs used by the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from typical_source_estimation.data import load_sequence_tsv
from typical_source_estimation.source_summary import summarize_source_sizes
from typical_source_estimation.tables import (
    DEFAULT_ALPHAS,
    emoji_topk_latex,
    estimator_bundle,
    markdown_table,
    source_size_summary_frame,
    topk_rows,
    write_text,
)


def build_outputs(
    *,
    input_path: str,
    outdir: str,
    source_col: str,
    sequence_col: str,
    sep: str,
    delimiter: str,
    min_freq: int,
    k: int,
) -> None:
    """Build all emoji Top-K reproduction outputs."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load the prepared WNS emoji TSV using the same delimiter-coded contract.
    dataset = load_sequence_tsv(
        input_path,
        source_col=source_col,
        sequence_col=sequence_col,
        delimiter=delimiter,
        sep=sep,
        min_freq=min_freq,
    )
    bundle = estimator_bundle(dataset, alphas=DEFAULT_ALPHAS)
    qhats = bundle.qhats()

    # Build Top-K and source-size tables from the exact same count matrix.
    topk_df = topk_rows(dataset, qhats, k=k)
    source_summary_df = source_size_summary_frame(summarize_source_sizes(dataset))

    # Write machine-readable and human-readable outputs side by side.
    topk_tsv = out_path / "emoji_topk_table.tsv"
    topk_md = out_path / "emoji_topk_table.md"
    topk_tex = out_path / "emoji_topk_table.tex"
    source_tsv = out_path / "emoji_source_size_summary.tsv"
    source_md = out_path / "emoji_source_size_summary.md"

    topk_df.to_csv(topk_tsv, sep="\t", index=False)
    source_summary_df.to_csv(source_tsv, sep="\t", index=False)
    write_text(topk_md, "# JADT emoji Top-K table\n\n" + markdown_table(topk_df) + "\n")
    write_text(source_md, "# JADT emoji source-size summary\n\n" + markdown_table(source_summary_df) + "\n")
    write_text(
        topk_tex,
        emoji_topk_latex(
            topk_df,
            caption=(
                "Top-20 emoji by estimator on the WNS corpus (probabilities rounded to 3~d.p.). "
                "Three values of $\alpha$ show the effect of relaxing the cap: smaller values give more "
                "source-balanced weights, while larger values allow high-mass sources to retain more influence. "
                "Head mass is the total probability assigned to the top-20 types. TV distances use the full "
                "distribution over all emoji types."
            ),
            label="tab:emoji",
        ),
    )

    # Print the key camera-ready diagnostics for command-line reproducibility.
    print(f"Loaded emoji dataset: S={dataset.S} sources, V={dataset.V} types, total tokens={int(dataset.total_mass)}")
    print(f"K={int(k)}")
    for name, q_hat in qhats.items():
        print(f"sum(q_hat[{name}]) = {float(q_hat.sum()):.12f}")
    print(f"Wrote: {topk_tsv}")
    print(f"Wrote: {topk_md}")
    print(f"Wrote: {topk_tex}")
    print(f"Wrote: {source_tsv}")
    print(f"Wrote: {source_md}")


def main() -> None:
    """Parse CLI arguments and generate the emoji reproduction outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Prepared emoji TSV with source/sequence columns")
    parser.add_argument("--outdir", default="outputs/emoji", help="Directory for generated outputs")
    parser.add_argument("--source-col", default="source_id", help="Source identifier column")
    parser.add_argument("--sequence-col", default="sequence", help="Emoji sequence column")
    parser.add_argument("--sep", default="\t", help="Input delimiter")
    parser.add_argument("--delimiter", default="-", help="Delimiter inside emoji sequences")
    parser.add_argument("--min-freq", type=int, default=1, help="Minimum token frequency")
    parser.add_argument("--k", type=int, default=20, help="Top-K size")
    args = parser.parse_args()

    build_outputs(
        input_path=str(args.input),
        outdir=str(args.outdir),
        source_col=str(args.source_col),
        sequence_col=str(args.sequence_col),
        sep=str(args.sep),
        delimiter=str(args.delimiter),
        min_freq=int(args.min_freq),
        k=int(args.k),
    )


if __name__ == "__main__":
    main()
