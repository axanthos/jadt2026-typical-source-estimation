"""Table-building helpers for the JADT reproduction scripts.

The functions in this module turn source-token count datasets into the compact
summary, Top-K, and shifted-token tables reported in the paper.  They contain no
WNS-specific input/output logic; scripts provide paths and call these helpers so
that the paper computations remain importable and unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from typical_source_estimation.data import CountsDataset, build_counts_dataset
from typical_source_estimation.estimators import EstimatorResult, capped_mass_alpha, pooled_mle, uniform_sources
from typical_source_estimation.metrics import total_variation
from typical_source_estimation.source_summary import SourceSizeSummary, summarize_source_sizes


DEFAULT_ALPHAS: tuple[float, ...] = (0.5, 1.0, 2.0)
DEFAULT_WORD_TOKEN_REGEX = r"(?u)\b\w+\b"
DEFAULT_LEXICAL_EXCLUDE_SOURCE_REGEX = r"(?i)^(#?wns\.system|#?system|system)$"


DEIDENTIFICATION_PLACEHOLDER_RE = re.compile(
    r"^_(last_name|url|email|number|town_name|street_address|masked_text)(?:_.*)?_$",
    re.IGNORECASE,
)
MEDIA_PLACEHOLDER_RE = re.compile(
    r"^_(audio|contact_card|document|gif|image|media|sticker|video)_omitted_$",
    re.IGNORECASE,
)
INTERACTION_PLACEHOLDER_RE = re.compile(
    r"^_(message_deleted|missed_(group_)?(voice|video)_call)_$",
    re.IGNORECASE,
)
SYSTEM_PLACEHOLDER_RE = re.compile(r"^__.+_message__$", re.IGNORECASE)
WNS_USER_PLACEHOLDER_RE = re.compile(r"^_wns_user_[0-9]{3}(?:_.*)?_$", re.IGNORECASE)
URL_LIKE_RE = re.compile(r"^(https?://|www\.)", re.IGNORECASE)
EMAIL_LIKE_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", re.IGNORECASE)
DATE_LIKE_RE = re.compile(r"^\d{1,4}([./:-]\d{1,4})+$")


@dataclass(frozen=True, slots=True)
class EstimatorBundle:
    """Named estimator results used by the paper table scripts."""

    results: dict[str, EstimatorResult]

    def qhats(self) -> dict[str, np.ndarray]:
        """Return estimator distributions keyed by display identifier."""
        return {name: result.q_hat for name, result in self.results.items()}


def estimator_bundle(dataset: CountsDataset, *, alphas: Sequence[float] = DEFAULT_ALPHAS) -> EstimatorBundle:
    """Compute the paper estimator family for a dataset."""
    results: dict[str, EstimatorResult] = {}

    # Compute the endpoint estimators first because table ordering depends on it.
    results["pooled"] = pooled_mle(dataset)
    results["unif_src"] = uniform_sources(dataset)

    # Add capped estimators under stable identifiers matching the paper outputs.
    for alpha in alphas:
        alpha_value = float(alpha)
        alpha_label = str(alpha_value).rstrip("0").rstrip(".")
        results[f"cap_a{alpha_label}"] = capped_mass_alpha(dataset, alpha=alpha_value)
    return EstimatorBundle(results=results)


def tv_between(qhats: Mapping[str, np.ndarray], first: str, second: str) -> float:
    """Return TV distance between two named estimated distributions."""
    try:
        return total_variation(qhats[first], qhats[second])
    except KeyError as exc:
        raise KeyError(f"unknown estimator in TV comparison: {exc.args[0]!r}") from exc


def source_size_summary_frame(summary: SourceSizeSummary) -> pd.DataFrame:
    """Return a source-size summary as a two-column display frame."""
    row = summary.as_dict()

    # Keep the display contract aligned with the camera-ready diagnostics.
    return pd.DataFrame(
        [
            {"metric": "sources", "value": f"{int(row['source_count']):,}"},
            {"metric": "tokens", "value": f"{int(row['total_tokens']):,}"},
            {
                "metric": "top 10% token share",
                "value": f"{100.0 * float(row['top_10pct_token_share']):.1f}% ({int(row['top_10pct_source_count'])} sources)",
            },
            {
                "metric": "bottom 50% token share",
                "value": f"{100.0 * float(row['bottom_50pct_token_share']):.1f}% ({int(row['bottom_50pct_source_count'])} sources)",
            },
            {"metric": "median tokens/source", "value": f"{float(row['median_tokens_per_source']):.0f}"},
            {"metric": "5th percentile tokens/source", "value": f"{float(row['p05_tokens_per_source']):.0f}"},
            {"metric": "max tokens/source", "value": f"{int(row['max_tokens_per_source']):,}"},
        ]
    )


def markdown_table(frame: pd.DataFrame) -> str:
    """Return a small GitHub-flavored Markdown table without optional deps.

    Pandas delegates ``DataFrame.to_markdown`` to the optional ``tabulate``
    package.  The reproduction package should not require an undeclared
    display-only dependency for script outputs, so this helper implements the
    compact pipe-table subset needed by our generated reports.
    """
    headers = [str(column) for column in frame.columns]
    body_rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]

    # Compute a deterministic width per column from headers and body cells.
    widths: list[int] = []
    for index, header in enumerate(headers):
        cell_lengths = [len(row[index]) for row in body_rows]
        widths.append(max([len(header), *cell_lengths]))

    # Keep alignment simple and stable for all generated documentation tables.
    def _format_row(cells: list[str]) -> str:
        padded = [cell.ljust(widths[index]) for index, cell in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    lines = [_format_row(headers), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(_format_row(row) for row in body_rows)
    return "\n".join(lines)


def latex_escape(value: object) -> str:
    """Escape a text value for use inside a LaTeX table cell."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    # Accented UTF-8 tokens are intentionally left unchanged for modern LaTeX.
    return "".join(replacements.get(ch, ch) for ch in str(value))


def write_text(path: str | Path, text: str) -> None:
    """Write UTF-8 text after creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def token_to_codepoints(token: str) -> str:
    """Return a hyphen-separated lowercase codepoint sequence for ``\twemoji``."""
    codepoints = [f"{ord(ch):x}" for ch in str(token)]

    # The twemojis package expects VS16 after gender signs in ZWJ sequences.
    patched: list[str] = []
    for index, code in enumerate(codepoints):
        patched.append(code)
        if code in {"2640", "2642"} and index > 0 and codepoints[index - 1] == "200d":
            if index + 1 >= len(codepoints) or codepoints[index + 1] != "fe0f":
                patched.append("fe0f")
    return "-".join(patched)


def twemoji(token: str) -> str:
    """Return a LaTeX ``\twemoji`` call for an emoji token."""
    return rf"\twemoji{{{token_to_codepoints(token)}}}"


def format_probability(value: float) -> str:
    """Format probability mass for compact emoji table cells."""
    return f"{float(value):.3f}"


def format_tv(value: float) -> str:
    """Format TV distances for camera-ready tables."""
    return f"{float(value):.3f}"


def format_delta(value: float) -> str:
    """Format signed per-10k lexical deltas."""
    return f"{float(value):+.2f}"


def artifact_reason(token: str) -> str | None:
    """Return the WNS artifact-filter reason for a lexical token, if any."""
    text = str(token).strip()
    lower = text.lower()

    # Apply documented WNS placeholders before generic technical artifacts.
    if DEIDENTIFICATION_PLACEHOLDER_RE.match(lower):
        return "deidentification_placeholder"
    if MEDIA_PLACEHOLDER_RE.match(lower):
        return "media_placeholder"
    if INTERACTION_PLACEHOLDER_RE.match(lower):
        return "interaction_placeholder"
    if SYSTEM_PLACEHOLDER_RE.match(lower):
        return "system_placeholder"
    if WNS_USER_PLACEHOLDER_RE.match(lower):
        return "wns_user_placeholder"

    # Drop residual technical forms that are not ordinary lexical evidence.
    if URL_LIKE_RE.match(lower):
        return "url_like_token"
    if EMAIL_LIKE_RE.match(lower):
        return "email_like_token"
    if DATE_LIKE_RE.match(lower):
        return "date_like_token"
    if lower.isdigit():
        return "numeric_token"
    if any(ch.isdigit() for ch in lower):
        return "digit_bearing_token"
    return None


def tokenize_text(value: object, *, token_regex: str = DEFAULT_WORD_TOKEN_REGEX) -> list[str]:
    """Lowercase and tokenize text using the paper's modest word regex."""
    if pd.isna(value):
        return []
    pattern = re.compile(token_regex)

    # Lowercase before matching to reproduce the camera-ready lexical check.
    return [match.group(0).lower() for match in pattern.finditer(str(value).lower())]


def load_word_tsv(
    path: str | Path,
    *,
    source_col: str = "source_id",
    text_col: str = "text",
    sep: str = "\t",
    min_freq: int = 1,
    token_regex: str = DEFAULT_WORD_TOKEN_REGEX,
    exclude_source_regex: str = DEFAULT_LEXICAL_EXCLUDE_SOURCE_REGEX,
    filter_artifacts: bool = True,
) -> CountsDataset:
    """Load a source/text TSV into a filtered lexical count dataset."""
    frame = pd.read_csv(path, sep=sep)
    missing = [col for col in (source_col, text_col) if col not in frame.columns]
    if missing:
        raise ValueError(f"missing required column(s) in {path}: {missing!r}")

    # Exclude system-like pseudo-sources before building source profiles.
    if str(exclude_source_regex or "").strip():
        pattern = re.compile(str(exclude_source_regex))
        frame = frame[~frame[source_col].astype(str).str.strip().str.match(pattern)]

    # Tokenize each row and apply the documented WNS lexical artifact filter.
    rows: list[tuple[str, list[str]]] = []
    for _, row in frame.iterrows():
        tokens = tokenize_text(row[text_col], token_regex=token_regex)
        if filter_artifacts:
            tokens = [token for token in tokens if artifact_reason(token) is None]
        rows.append((str(row[source_col]), tokens))

    return build_counts_dataset(
        rows,
        min_freq=min_freq,
        metadata={
            "path": str(path),
            "source_col": source_col,
            "text_col": text_col,
            "sep": sep,
            "min_freq": int(min_freq),
            "token_regex": token_regex,
            "filter_artifacts": bool(filter_artifacts),
            "exclude_source_regex": exclude_source_regex,
        },
    )


def topk_rows(dataset: CountsDataset, qhats: Mapping[str, np.ndarray], *, k: int) -> pd.DataFrame:
    """Build the multi-estimator Top-K table used for emoji output."""
    columns = _emoji_display_columns(qhats)
    top_indices: dict[str, np.ndarray] = {}

    # Sort each estimator distribution by probability, then by token string.
    for name in columns:
        q_hat = qhats[name]
        order = sorted(range(dataset.V), key=lambda idx: (-float(q_hat[idx]), dataset.vocab[idx]))
        top_indices[name] = np.asarray(order[: int(k)], dtype=int)

    # Emit one display row per rank with token and probability in each cell.
    rows: list[dict[str, object]] = []
    for rank in range(int(k)):
        row: dict[str, object] = {"rank": rank + 1}
        for name in columns:
            idx = int(top_indices[name][rank])
            row[name] = f"{dataset.vocab[idx]} {format_probability(float(qhats[name][idx]))}"
        rows.append(row)

    # Append compact summary rows following the paper-table contract.
    summary_rows = _emoji_summary_rows(dataset, qhats, top_indices, columns)
    return pd.DataFrame(rows + summary_rows)


def _emoji_display_columns(qhats: Mapping[str, np.ndarray]) -> list[str]:
    """Return the camera-ready emoji table order, with POOL at the right."""
    preferred = ["unif_src", "cap_a0.5", "cap_a1", "cap_a2", "pooled"]
    ordered = [name for name in preferred if name in qhats]

    # Preserve any additional estimators after the paper columns for extensibility.
    ordered.extend(name for name in qhats.keys() if name not in ordered)
    return ordered


def _emoji_summary_rows(
    dataset: CountsDataset,
    qhats: Mapping[str, np.ndarray],
    top_indices: Mapping[str, np.ndarray],
    columns: Sequence[str],
) -> list[dict[str, object]]:
    """Return compact full-distribution diagnostics for the Top-K table."""
    rows: list[dict[str, object]] = []

    # Head mass is computed on each estimator's own top-K set.
    mass_row: dict[str, object] = {"rank": "Head mass"}
    for name in columns:
        mass_row[name] = format_probability(float(np.asarray(qhats[name])[top_indices[name]].sum()))
    rows.append(mass_row)

    # TV rows use the full distribution over the complete emoji vocabulary.
    for reference, label in (("pooled", "TV to POOL"), ("unif_src", "TV to UNIF")):
        row: dict[str, object] = {"rank": label}
        for name in columns:
            if name == reference:
                row[name] = "--"
            else:
                row[name] = format_probability(tv_between(qhats, reference, name))
        rows.append(row)
    return rows


def emoji_topk_latex(table_df: pd.DataFrame, *, caption: str, label: str) -> str:
    """Return a LaTeX table for the emoji Top-K output."""
    columns = [col for col in table_df.columns if col != "rank"]
    lines: list[str] = [
        "% Auto-generated by scripts/reproduce_emoji_table.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
    ]

    # Use the compact camera-ready header when the paper estimator order is present.
    if columns == ["unif_src", "cap_a0.5", "cap_a1", "cap_a2", "pooled"]:
        lines.extend(
            [
                r"\begin{tabular}{r l l l l l}",
                r"\hline",
                r" & & \multicolumn{3}{c}{\textsc{cap}} & \\",
                r"\cline{3-5}",
                r"Rank & \textsc{unif} & $\alpha$=0.5 & $\alpha$=1.0 & $\alpha$=2.0 & \textsc{pool} \\",
                r"\hline",
            ]
        )
    else:
        lines.extend(
            [
                r"\begin{tabular}{r" + "l" * len(columns) + "}",
                r"\hline",
                "Rank & " + " & ".join(_emoji_header(col) for col in columns) + r" \\",
                r"\hline",
            ]
        )

    # Render emoji cells with \twemoji and numeric summary cells as plain text.
    for _, row in table_df.iterrows():
        cells = [_emoji_rank_latex(row["rank"])]
        for col in columns:
            cells.append(_emoji_latex_cell(row[col]))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{latex_escape(label)}}}", r"\end{table}", ""])
    return "\n".join(lines)


def _emoji_rank_latex(value: object) -> str:
    """Render rank or summary-row labels for the LaTeX emoji table."""
    text = str(value)
    if text == "Head mass":
        return "Head mass"
    if text == "TV to POOL":
        return r"TV to \textsc{pool}"
    if text == "TV to UNIF":
        return r"TV to \textsc{unif}"
    return latex_escape(text)


def _emoji_header(name: str) -> str:
    """Return the LaTeX column header for an estimator identifier."""
    if name == "pooled":
        return r"\textsc{pool}"
    if name == "unif_src":
        return r"\textsc{unif}"
    if name.startswith("cap_a"):
        alpha = name.removeprefix("cap_a")
        return rf"\textsc{{cap}} $\alpha={latex_escape(alpha)}$"
    return latex_escape(name)


def _emoji_latex_cell(value: object) -> str:
    """Render one emoji table cell, preserving summary rows as plain text."""
    text = str(value)
    if " " not in text:
        return latex_escape(text)
    token, probability = text.rsplit(" ", 1)
    try:
        float(probability)
    except ValueError:
        return latex_escape(text)
    return f"{twemoji(token)}{{}}~{latex_escape(probability)}"


def lexical_tv_summary(qhats: Mapping[str, np.ndarray], *, contrast: str = "cap_a1") -> pd.DataFrame:
    """Return the compact lexical TV matrix in POOL--CAP--UNIF order."""
    return pd.DataFrame(
        [
            {"": "POOL", "POOL": "--", "CAP": format_tv(tv_between(qhats, "pooled", contrast)), "UNIF": format_tv(tv_between(qhats, "pooled", "unif_src"))},
            {"": "CAP", "POOL": format_tv(tv_between(qhats, "pooled", contrast)), "CAP": "--", "UNIF": format_tv(tv_between(qhats, contrast, "unif_src"))},
            {"": "UNIF", "POOL": format_tv(tv_between(qhats, "pooled", "unif_src")), "CAP": format_tv(tv_between(qhats, contrast, "unif_src")), "UNIF": "--"},
        ]
    )


def full_shift_frame(dataset: CountsDataset, qhats: Mapping[str, np.ndarray], *, contrast: str) -> pd.DataFrame:
    """Return per-token lexical shift diagnostics for a contrast estimator."""
    pooled = np.asarray(qhats["pooled"], dtype=np.float64)
    contrast_q = np.asarray(qhats[contrast], dtype=np.float64)
    unif = np.asarray(qhats["unif_src"], dtype=np.float64)
    source_counts = np.asarray((dataset.counts > 0).sum(axis=0)).reshape(-1)
    totals = np.asarray(dataset.counts.sum(axis=0)).reshape(-1)

    # Report per-token frequencies on the interpretable per-10k scale.
    rows: list[dict[str, object]] = []
    for idx, token in enumerate(dataset.vocab):
        delta = float(contrast_q[idx] - pooled[idx])
        rows.append(
            {
                "token": token,
                "total_count": int(totals[idx]),
                "source_count": int(source_counts[idx]),
                "pooled_per_10k": 10000.0 * float(pooled[idx]),
                "unif_src_per_10k": 10000.0 * float(unif[idx]),
                f"{contrast}_per_10k": 10000.0 * float(contrast_q[idx]),
                "delta_unif_src_minus_pooled_per_10k": 10000.0 * float(unif[idx] - pooled[idx]),
                f"delta_{contrast}_minus_pooled_per_10k": 10000.0 * delta,
                f"tv_contrib_{contrast}_vs_pooled_per_10k": 0.5 * 10000.0 * abs(delta),
            }
        )
    return pd.DataFrame(rows)


def lexical_shift_table(full_df: pd.DataFrame, *, contrast: str = "cap_a1", k: int = 10) -> pd.DataFrame:
    """Return the two-sided signed lexical shift table."""
    delta_col = f"delta_{contrast}_minus_pooled_per_10k"
    positive = full_df[full_df[delta_col] > 0].sort_values([delta_col, "token"], ascending=[False, True]).head(k).reset_index(drop=True)
    negative = full_df[full_df[delta_col] < 0].sort_values([delta_col, "token"], ascending=[True, True]).head(k).reset_index(drop=True)

    # Pair positive and negative shifts row-wise for compact paper display.
    rows: list[dict[str, object]] = []
    for idx in range(max(len(positive), len(negative))):
        pos = positive.iloc[idx] if idx < len(positive) else None
        neg = negative.iloc[idx] if idx < len(negative) else None
        rows.append(
            {
                "higher_under_cap": "" if pos is None else pos["token"],
                "cap_source_count": np.nan if pos is None else int(pos["source_count"]),
                "cap_delta_per_10k": np.nan if pos is None else float(pos[delta_col]),
                "higher_under_pool": "" if neg is None else neg["token"],
                "pool_source_count": np.nan if neg is None else int(neg["source_count"]),
                "pool_delta_per_10k": np.nan if neg is None else float(neg[delta_col]),
            }
        )
    return pd.DataFrame(rows)


def lexical_tables_latex(tv_df: pd.DataFrame, shift_df: pd.DataFrame, *, contrast_label: str = "CAP") -> str:
    """Return LaTeX table environments for the lexical sensitivity output."""
    lines: list[str] = [
        "% Auto-generated by scripts/reproduce_lexical_tables.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r" & \textsc{pool} & \textsc{cap} ($\alpha=1$) & \textsc{unif} \\",
        r"\hline",
    ]

    # The compact distance matrix is formatted directly from the display frame.
    for _, row in tv_df.iterrows():
        label = str(row[""]).lower()
        if label == "cap":
            left = r"\textsc{cap} ($\alpha=1$)"
        else:
            left = rf"\textsc{{{label}}}"
        lines.append(f"{left} & {row['POOL']} & {row['CAP']} & {row['UNIF']} " + r"\\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Total variation distances between lexical estimates after excluding documented WNS placeholders and simple technical artifacts.}",
            r"\label{tab:lexical-tv}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrrlrr}",
            r"\hline",
            rf"Higher under {latex_escape(contrast_label)} & $n_{{\mathrm{{src}}}}$ & $\Delta$/10k & Higher under POOL & $n_{{\mathrm{{src}}}}$ & $\Delta$/10k \\",
            r"\hline",
        ]
    )

    # Render lexical tokens as escaped text but keep the LaTeX caption raw.
    for _, row in shift_df.iterrows():
        lines.append(
            "{} & {} & {} & {} & {} & {} \\".format(
                latex_escape(row["higher_under_cap"]),
                "" if pd.isna(row["cap_source_count"]) else int(row["cap_source_count"]),
                "" if pd.isna(row["cap_delta_per_10k"]) else latex_escape(format_delta(row["cap_delta_per_10k"])),
                latex_escape(row["higher_under_pool"]),
                "" if pd.isna(row["pool_source_count"]) else int(row["pool_source_count"]),
                "" if pd.isna(row["pool_delta_per_10k"]) else latex_escape(format_delta(row["pool_delta_per_10k"])),
            )
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Largest signed lexical shifts between the capped ($\alpha=1$) and pooled estimates. Values are differences per 10,000 tokens; $n_{\mathrm{src}}$ gives the number of sources in which the token occurs.}",
            r"\label{tab:lexical-shifts}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)
