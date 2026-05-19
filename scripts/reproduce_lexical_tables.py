"""Reproduce the JADT lexical sensitivity tables.

The script consumes the two-column lexical text-run TSV generated locally from
the controlled-access WNS corpus.  It applies the paper's modest lowercasing,
word-tokenization, and WNS placeholder/artifact filtering before computing the
lexical TV matrix and signed-shift table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from typical_source_estimation.source_summary import summarize_source_sizes
from typical_source_estimation.tables import (
    DEFAULT_LEXICAL_EXCLUDE_SOURCE_REGEX,
    DEFAULT_WORD_TOKEN_REGEX,
    estimator_bundle,
    full_shift_frame,
    lexical_shift_table,
    lexical_tables_latex,
    lexical_tv_summary,
    load_word_tsv,
    markdown_table,
    source_size_summary_frame,
    write_text,
)


def build_outputs(
    *,
    input_path: str,
    outdir: str,
    source_col: str,
    text_col: str,
    sep: str,
    min_freq: int,
    token_regex: str,
    exclude_source_regex: str,
    contrast: str,
    k: int,
) -> None:
    """Build all lexical sensitivity reproduction outputs."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load and filter the prepared WNS lexical TSV exactly once.
    dataset = load_word_tsv(
        input_path,
        source_col=source_col,
        text_col=text_col,
        sep=sep,
        min_freq=min_freq,
        token_regex=token_regex,
        exclude_source_regex=exclude_source_regex,
        filter_artifacts=True,
    )
    qhats = estimator_bundle(dataset).qhats()

    # Build compact paper-facing TV and signed-shift tables.
    tv_df = lexical_tv_summary(qhats, contrast=contrast)
    full_df = full_shift_frame(dataset, qhats, contrast=contrast)
    shift_df = lexical_shift_table(full_df, contrast=contrast, k=k)
    source_summary_df = source_size_summary_frame(summarize_source_sizes(dataset))

    # Write all outputs so readers can inspect both display and full diagnostics.
    tv_tsv = out_path / "lexical_tv_table.tsv"
    shift_tsv = out_path / "lexical_shift_table.tsv"
    full_tsv = out_path / "lexical_full_shift_table.tsv"
    source_tsv = out_path / "lexical_source_size_summary.tsv"
    md_path = out_path / "lexical_tables.md"
    tex_path = out_path / "lexical_tables.tex"

    tv_df.to_csv(tv_tsv, sep="\t", index=False)
    shift_df.to_csv(shift_tsv, sep="\t", index=False)
    full_df.to_csv(full_tsv, sep="\t", index=False)
    source_summary_df.to_csv(source_tsv, sep="\t", index=False)
    write_text(md_path, _markdown_report(tv_df, shift_df, source_summary_df))
    write_text(tex_path, lexical_tables_latex(tv_df, shift_df, contrast_label="CAP"))

    # Print the key diagnostics used in the camera-ready prose.
    pool_cap = str(tv_df.loc[tv_df[""] == "POOL", "CAP"].iloc[0])
    pool_unif = str(tv_df.loc[tv_df[""] == "POOL", "UNIF"].iloc[0])
    cap_unif = str(tv_df.loc[tv_df[""] == "CAP", "UNIF"].iloc[0])
    print(f"Filtered lexical dataset: S={dataset.S} sources, V={dataset.V} types, total tokens={int(dataset.total_mass)}")
    print(f"TV summary: TV(POOL, UNIF)={pool_unif}, TV(POOL, CAP)={pool_cap}, TV(UNIF, CAP)={cap_unif}")
    print(f"Wrote: {tv_tsv}")
    print(f"Wrote: {shift_tsv}")
    print(f"Wrote: {full_tsv}")
    print(f"Wrote: {source_tsv}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {tex_path}")


def _markdown_report(tv_df, shift_df, source_summary_df) -> str:
    """Return the Markdown report for lexical reproduction outputs."""
    return "\n".join(
        [
            "# JADT lexical estimator-sensitivity tables",
            "",
            "## TV distance matrix",
            "",
            markdown_table(tv_df),
            "",
            "## Source-size summary",
            "",
            markdown_table(source_summary_df),
            "",
            "## Shifted tokens",
            "",
            markdown_table(_shift_markdown_frame(shift_df)),
            "",
        ]
    )


def _shift_markdown_frame(shift_df):
    """Return a display-friendly lexical shift frame for Markdown output."""
    rows: list[dict[str, object]] = []
    for _, row in shift_df.iterrows():
        rows.append(
            {
                "higher under CAP": row["higher_under_cap"],
                "n src.": "" if pd.isna(row["cap_source_count"]) else int(row["cap_source_count"]),
                "Δ/10k": "" if pd.isna(row["cap_delta_per_10k"]) else f"{float(row['cap_delta_per_10k']):+.2f}",
                "higher under POOL": row["higher_under_pool"],
                "n src. ": "" if pd.isna(row["pool_source_count"]) else int(row["pool_source_count"]),
                "Δ/10k ": "" if pd.isna(row["pool_delta_per_10k"]) else f"{float(row['pool_delta_per_10k']):+.2f}",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Parse CLI arguments and generate lexical reproduction outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Prepared lexical TSV with source/text columns")
    parser.add_argument("--outdir", default="outputs/lexical", help="Directory for generated outputs")
    parser.add_argument("--source-col", default="source_id", help="Source identifier column")
    parser.add_argument("--text-col", default="text", help="Text-run column")
    parser.add_argument("--sep", default="\t", help="Input delimiter")
    parser.add_argument("--min-freq", type=int, default=1, help="Minimum word-token frequency")
    parser.add_argument("--token-regex", default=DEFAULT_WORD_TOKEN_REGEX, help="Word-token regex")
    parser.add_argument("--exclude-source-regex", default=DEFAULT_LEXICAL_EXCLUDE_SOURCE_REGEX, help="System source exclusion regex")
    parser.add_argument("--contrast", default="cap_a1", help="Contrast estimator for shifted-token ranking")
    parser.add_argument("--k", type=int, default=10, help="Tokens per signed-shift side")
    args = parser.parse_args()

    build_outputs(
        input_path=str(args.input),
        outdir=str(args.outdir),
        source_col=str(args.source_col),
        text_col=str(args.text_col),
        sep=str(args.sep),
        min_freq=int(args.min_freq),
        token_regex=str(args.token_regex),
        exclude_source_regex=str(args.exclude_source_regex),
        contrast=str(args.contrast),
        k=int(args.k),
    )


if __name__ == "__main__":
    main()
