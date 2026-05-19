#!/usr/bin/env python
"""Compare legacy and reproduced WNS lexical input TSVs.

The paper's lexical check was first run from a rich WNS-derived table, while
this reproduction package generates a minimal two-column lexical TSV from the
controlled-access XML corpus.  This diagnostic projects both inputs through the
same source filter, tokenizer, and lexical artifact filter so that differences
can be inspected at the source-count and token-count levels.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from typical_source_estimation.tables import (
    DEFAULT_LEXICAL_EXCLUDE_SOURCE_REGEX,
    DEFAULT_WORD_TOKEN_REGEX,
    artifact_reason,
    markdown_table,
    tokenize_text,
    write_text,
)


@dataclass(frozen=True, slots=True)
class LexicalInputStats:
    """Token-count summaries for one lexical input table."""

    label: str
    input_rows: int
    rows_after_source_filter: int
    raw_source_tokens: dict[str, int]
    filtered_source_tokens: dict[str, int]
    raw_token_counts: Counter[str]
    filtered_token_counts: Counter[str]

    @property
    def raw_token_total(self) -> int:
        """Return the total raw token count after source filtering."""
        return int(sum(self.raw_token_counts.values()))

    @property
    def filtered_token_total(self) -> int:
        """Return the total filtered token count after source filtering."""
        return int(sum(self.filtered_token_counts.values()))

    @property
    def raw_vocab_size(self) -> int:
        """Return the number of raw token types after source filtering."""
        return int(len(self.raw_token_counts))

    @property
    def filtered_vocab_size(self) -> int:
        """Return the number of filtered token types after source filtering."""
        return int(len(self.filtered_token_counts))


def _read_tsv(path: str | Path, *, sep: str) -> pd.DataFrame:
    """Read a TSV while preserving empty strings and embedded newlines."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input TSV does not exist: {source}")

    # Preserve empty metadata cells as empty strings rather than NaN values.
    return pd.read_csv(source, sep=sep, keep_default_na=False)


def _source_is_excluded(source_id: object, pattern: re.Pattern[str] | None) -> bool:
    """Return whether a source identifier is excluded from analysis."""
    if pattern is None:
        return False
    return bool(pattern.match(str(source_id).strip()))


def collect_stats(
    *,
    path: str | Path,
    label: str,
    source_col: str,
    text_col: str,
    sep: str,
    token_regex: str,
    exclude_source_regex: str,
) -> LexicalInputStats:
    """Collect raw and artifact-filtered lexical token counts for one TSV."""
    frame = _read_tsv(path, sep=sep)
    missing = [column for column in (source_col, text_col) if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required column(s) in {path}: {missing!r}")

    # Compile the source filter once; an empty pattern disables exclusion.
    pattern_text = str(exclude_source_regex or "").strip()
    exclude_pattern = re.compile(pattern_text) if pattern_text else None

    # Accumulate source-level and token-level counts in parallel.
    raw_source_tokens: defaultdict[str, int] = defaultdict(int)
    filtered_source_tokens: defaultdict[str, int] = defaultdict(int)
    raw_token_counts: Counter[str] = Counter()
    filtered_token_counts: Counter[str] = Counter()
    kept_rows = 0

    for _, row in frame.iterrows():
        source_id = str(row[source_col])
        if _source_is_excluded(source_id, exclude_pattern):
            continue
        kept_rows += 1

        # Tokenize with the same modest regex used by the paper table script.
        raw_tokens = tokenize_text(row[text_col], token_regex=token_regex)
        filtered_tokens = [token for token in raw_tokens if artifact_reason(token) is None]

        # Update the source and vocabulary counts for raw and filtered views.
        raw_source_tokens[source_id] += len(raw_tokens)
        filtered_source_tokens[source_id] += len(filtered_tokens)
        raw_token_counts.update(raw_tokens)
        filtered_token_counts.update(filtered_tokens)

    return LexicalInputStats(
        label=label,
        input_rows=int(len(frame)),
        rows_after_source_filter=int(kept_rows),
        raw_source_tokens=dict(raw_source_tokens),
        filtered_source_tokens=dict(filtered_source_tokens),
        raw_token_counts=raw_token_counts,
        filtered_token_counts=filtered_token_counts,
    )


def _source_delta_frame(old: LexicalInputStats, new: LexicalInputStats) -> pd.DataFrame:
    """Return source-level raw and filtered token-count deltas."""
    source_ids = sorted(set(old.raw_source_tokens) | set(new.raw_source_tokens) | set(old.filtered_source_tokens) | set(new.filtered_source_tokens))
    rows: list[dict[str, object]] = []

    for source_id in source_ids:
        old_raw = int(old.raw_source_tokens.get(source_id, 0))
        new_raw = int(new.raw_source_tokens.get(source_id, 0))
        old_filtered = int(old.filtered_source_tokens.get(source_id, 0))
        new_filtered = int(new.filtered_source_tokens.get(source_id, 0))
        filtered_delta = new_filtered - old_filtered
        rel_delta = "" if old_filtered == 0 else f"{filtered_delta / old_filtered:.6f}"

        # Store both raw and filtered deltas so concentration is easy to inspect.
        rows.append(
            {
                "source_id": source_id,
                "old_raw_tokens": old_raw,
                "new_raw_tokens": new_raw,
                "raw_delta": new_raw - old_raw,
                "old_filtered_tokens": old_filtered,
                "new_filtered_tokens": new_filtered,
                "filtered_delta": filtered_delta,
                "filtered_rel_delta_vs_old": rel_delta,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["filtered_delta", "raw_delta", "source_id"], key=_sort_abs_except_id, ascending=[False, False, True])


def _sort_abs_except_id(series: pd.Series) -> pd.Series:
    """Sort numeric delta columns by absolute value while leaving labels intact."""
    if pd.api.types.is_numeric_dtype(series):
        return series.abs()
    return series


def _token_delta_frame(old_counts: Counter[str], new_counts: Counter[str], *, include_reason: bool) -> pd.DataFrame:
    """Return token-level count deltas for raw or filtered vocabularies."""
    rows: list[dict[str, object]] = []
    for token in sorted(set(old_counts) | set(new_counts)):
        old_count = int(old_counts.get(token, 0))
        new_count = int(new_counts.get(token, 0))
        delta = new_count - old_count
        if delta == 0:
            continue

        # Include artifact reasons for raw counts; filtered tokens should have none.
        row: dict[str, object] = {
            "token": token,
            "old_count": old_count,
            "new_count": new_count,
            "delta": delta,
        }
        if include_reason:
            row["artifact_reason"] = artifact_reason(token) or ""
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["delta", "token"], key=_sort_abs_except_id, ascending=[False, True])


def _global_summary_frame(old: LexicalInputStats, new: LexicalInputStats) -> pd.DataFrame:
    """Return a compact old/new global summary frame."""
    metrics = [
        ("input_rows", old.input_rows, new.input_rows),
        ("rows_after_source_filter", old.rows_after_source_filter, new.rows_after_source_filter),
        ("raw_sources_with_tokens", len(old.raw_source_tokens), len(new.raw_source_tokens)),
        ("filtered_sources_with_tokens", len(old.filtered_source_tokens), len(new.filtered_source_tokens)),
        ("raw_token_count", old.raw_token_total, new.raw_token_total),
        ("filtered_token_count", old.filtered_token_total, new.filtered_token_total),
        ("raw_vocab_types", old.raw_vocab_size, new.raw_vocab_size),
        ("filtered_vocab_types", old.filtered_vocab_size, new.filtered_vocab_size),
    ]

    # Emit integer deltas to make the offset visible in the first diagnostic file.
    return pd.DataFrame(
        [
            {"metric": name, "old": int(old_value), "new": int(new_value), "delta_new_minus_old": int(new_value) - int(old_value)}
            for name, old_value, new_value in metrics
        ]
    )


def _write_frame_pair(path_tsv: Path, path_md: Path, frame: pd.DataFrame, *, md_rows: int | None = None) -> None:
    """Write a diagnostic frame as TSV and Markdown."""
    path_tsv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path_tsv, sep="\t", index=False)

    # Markdown can be limited to the most relevant rows for readability.
    display = frame if md_rows is None else frame.head(int(md_rows))
    write_text(path_md, markdown_table(display))


def build_outputs(
    *,
    old_path: str,
    new_path: str,
    outdir: str,
    old_source_col: str,
    old_text_col: str,
    new_source_col: str,
    new_text_col: str,
    sep: str,
    token_regex: str,
    exclude_source_regex: str,
    markdown_rows: int,
) -> None:
    """Build global, source-level, and token-level lexical comparison outputs."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Project both inputs into the same tokenization/filtering contract.
    old = collect_stats(
        path=old_path,
        label="old",
        source_col=old_source_col,
        text_col=old_text_col,
        sep=sep,
        token_regex=token_regex,
        exclude_source_regex=exclude_source_regex,
    )
    new = collect_stats(
        path=new_path,
        label="new",
        source_col=new_source_col,
        text_col=new_text_col,
        sep=sep,
        token_regex=token_regex,
        exclude_source_regex=exclude_source_regex,
    )

    # Build the comparison tables at global, source, and token levels.
    global_summary = _global_summary_frame(old, new)
    source_deltas = _source_delta_frame(old, new)
    filtered_token_deltas = _token_delta_frame(old.filtered_token_counts, new.filtered_token_counts, include_reason=False)
    raw_token_deltas = _token_delta_frame(old.raw_token_counts, new.raw_token_counts, include_reason=True)

    # Write full TSVs and compact Markdown previews for triage in RF/browser views.
    _write_frame_pair(out_path / "global_summary.tsv", out_path / "global_summary.md", global_summary)
    _write_frame_pair(out_path / "source_token_deltas.tsv", out_path / "source_token_deltas.md", source_deltas, md_rows=markdown_rows)
    _write_frame_pair(
        out_path / "token_count_deltas_filtered.tsv",
        out_path / "token_count_deltas_filtered.md",
        filtered_token_deltas,
        md_rows=markdown_rows,
    )
    _write_frame_pair(out_path / "token_count_deltas_raw.tsv", out_path / "token_count_deltas_raw.md", raw_token_deltas, md_rows=markdown_rows)

    print("Lexical input comparison")
    print(f"  old: {old_path}")
    print(f"  new: {new_path}")
    print(
        "  filtered tokens: "
        f"old={old.filtered_token_total:,}, new={new.filtered_token_total:,}, "
        f"delta={new.filtered_token_total - old.filtered_token_total:+,}"
    )
    print(
        "  filtered vocabulary: "
        f"old={old.filtered_vocab_size:,}, new={new.filtered_vocab_size:,}, "
        f"delta={new.filtered_vocab_size - old.filtered_vocab_size:+,}"
    )
    print(f"Wrote: {out_path / 'global_summary.tsv'}")
    print(f"Wrote: {out_path / 'source_token_deltas.tsv'}")
    print(f"Wrote: {out_path / 'token_count_deltas_filtered.tsv'}")
    print(f"Wrote: {out_path / 'token_count_deltas_raw.tsv'}")


def main() -> None:
    """Parse command-line arguments and write lexical comparison outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", default="data/private/wns_corpus_v1.0.0_no_EMJ.tsv", help="Legacy/rich lexical TSV used by the earlier paper pipeline.")
    parser.add_argument("--new", default="data/private/derived/wns_lexical.tsv", help="New minimal lexical TSV generated by this reproduction package.")
    parser.add_argument("--outdir", default="outputs/lexical_compare", help="Output directory for comparison diagnostics.")
    parser.add_argument("--old-source-col", default="who", help="Source column in the legacy TSV.")
    parser.add_argument("--old-text-col", default="string", help="Text column in the legacy TSV.")
    parser.add_argument("--new-source-col", default="source_id", help="Source column in the new TSV.")
    parser.add_argument("--new-text-col", default="text", help="Text column in the new TSV.")
    parser.add_argument("--sep", default="\t", help="Input and output delimiter for TSV files.")
    parser.add_argument("--token-regex", default=DEFAULT_WORD_TOKEN_REGEX, help="Regex used for lowercase word-token extraction.")
    parser.add_argument("--exclude-source-regex", default=DEFAULT_LEXICAL_EXCLUDE_SOURCE_REGEX, help="Regex for source identifiers excluded before comparison.")
    parser.add_argument("--markdown-rows", type=int, default=50, help="Rows to include in Markdown previews for long delta tables.")
    args = parser.parse_args()

    build_outputs(
        old_path=str(args.old),
        new_path=str(args.new),
        outdir=str(args.outdir),
        old_source_col=str(args.old_source_col),
        old_text_col=str(args.old_text_col),
        new_source_col=str(args.new_source_col),
        new_text_col=str(args.new_text_col),
        sep=str(args.sep),
        token_regex=str(args.token_regex),
        exclude_source_regex=str(args.exclude_source_regex),
        markdown_rows=int(args.markdown_rows),
    )


if __name__ == "__main__":
    main()
