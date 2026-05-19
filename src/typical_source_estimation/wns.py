"""WNS preparation helpers for controlled-access local reproduction.

The public package does not redistribute token-level material from *What's New,
Switzerland?* (WNS).  Authorized users run the helpers in this module against a
local controlled-access copy of the XML-TEI corpus to regenerate the TSV inputs
used by the paper's real-data analyses.

The emoji and text-run extraction code is a cleaned, package-level port of the
WNS utility script used to prepare the JADT inputs.  It intentionally relies on
the :mod:`emoji` package for emoji segmentation, rather than approximating the
Unicode emoji specification with local code.
"""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List

import emoji
from emoji.tokenizer import EmojiMatch


XML_NS = "http://www.w3.org/XML/1998/namespace"
XML_ID = f"{{{XML_NS}}}id"
ZWJ = "\u200d"
VS16 = "\ufe0f"
VS15 = "\ufe0e"
KEYCAP = "\u20e3"
GENDER_SIGNS = {"\u2640", "\u2642"}
SKIN_TONE_MIN = 0x1F3FB
SKIN_TONE_MAX = 0x1F3FF
SYSTEM_SOURCE_RE = re.compile(r"(?i)^(#?wns\.system|#?system|system)$")


@dataclass(frozen=True, slots=True)
class WnsPost:
    """Minimal post record extracted from a WNS XML-TEI file.

    The fields are limited to what is needed to regenerate the paper's emoji and
    lexical TSV inputs.  Richer WNS metadata remain available through the
    controlled-access corpus itself rather than this reproduction package.
    """

    source_file: str
    post_id: str
    source_id: str
    generated_by: str
    modality: str
    text: str


def _local_name(tag: str) -> str:
    """Return the local XML name independently of namespace decoration."""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _xml_id(element: ET.Element) -> str:
    """Return an element's ``xml:id`` value when present."""
    return str(element.attrib.get(XML_ID) or element.attrib.get("xml:id") or "")


def _post_text_without_time(post: ET.Element) -> str:
    """Return post text while excluding the timestamp element's own text."""
    pieces: list[str] = []

    def collect(element: ET.Element) -> None:
        """Append text recursively, skipping content inside ``<time>`` nodes."""
        if _local_name(element.tag) != "time" and element.text:
            pieces.append(element.text)

        # Child tails belong to the surrounding post text even for ``<time>``.
        for child in list(element):
            if _local_name(child.tag) != "time":
                collect(child)
            if child.tail:
                pieces.append(child.tail)

    # Match the WNS exporter contract: concatenate non-time text and strip it.
    collect(post)
    return "".join(pieces).strip()


def _iter_xml_files(xml_dir: str | Path, pattern: str) -> list[Path]:
    """Return sorted WNS XML files, failing early when none are found."""
    folder = Path(xml_dir)
    files = sorted(folder.glob(pattern))

    # A missing input directory is usually a path/configuration error.
    if not folder.exists():
        raise FileNotFoundError(f"WNS XML directory does not exist: {folder}")
    if not files:
        raise FileNotFoundError(f"No WNS XML files found in {folder} matching {pattern!r}.")
    return files


def iter_wns_posts(xml_dir: str | Path, *, pattern: str = "wns_chat_*.xml") -> Iterator[WnsPost]:
    """Yield minimal WNS post records from XML-TEI files."""
    for path in _iter_xml_files(xml_dir, pattern):
        root = ET.parse(path).getroot()

        # Search by local name so the function works with namespaced TEI files.
        for post in root.iter():
            if _local_name(post.tag) != "post":
                continue
            yield WnsPost(
                source_file=path.name,
                post_id=_xml_id(post),
                source_id=str(post.attrib.get("who", "")),
                generated_by=str(post.attrib.get("generatedBy", "")),
                modality=str(post.attrib.get("modality", "")),
                text=_post_text_without_time(post),
            )


def write_posts_tsv(
    posts: Iterable[WnsPost],
    output_path: str | Path,
    *,
    sep: str = "\t",
) -> None:
    """Write a minimal post-level TSV used by downstream preparation scripts."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["source_file", "post_id", "source_id", "generated_by", "modality", "text"]

    # Use the standard CSV writer to keep quoting and line endings explicit.
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter=sep, lineterminator="\n")
        writer.writeheader()
        for post in posts:
            writer.writerow({column: getattr(post, column) for column in columns})


def prepare_posts_tsv(
    xml_dir: str | Path,
    output_path: str | Path,
    *,
    pattern: str = "wns_chat_*.xml",
    sep: str = "\t",
) -> None:
    """Extract WNS XML posts into the package's minimal post-level TSV."""
    write_posts_tsv(iter_wns_posts(xml_dir, pattern=pattern), output_path, sep=sep)


def _strip_variation_selectors(text: str) -> str:
    """Remove Unicode emoji/text presentation selectors from ``text``."""
    return text.replace(VS16, "").replace(VS15, "")


def _strip_skin_tone(text: str) -> str:
    """Remove Fitzpatrick skin-tone modifiers from ``text``."""
    return "".join(ch for ch in text if not (SKIN_TONE_MIN <= ord(ch) <= SKIN_TONE_MAX))


def _strip_gender(text: str) -> str:
    """Remove gender signs and their immediately following variation selector."""
    out: list[str] = []
    i = 0

    # Keep the original WNS utility behavior: when a gender sign is stripped,
    # consume an attached VS immediately after it as part of the same marker.
    while i < len(text):
        ch = text[i]
        if ch in GENDER_SIGNS:
            i += 1
            if i < len(text) and text[i] in {VS16, VS15}:
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _is_tag_char(ch: str) -> bool:
    """Return whether ``ch`` is a Unicode tag character used in some emoji."""
    code = ord(ch)
    return 0xE0020 <= code <= 0xE007F


def _is_base_for_zwj(ch: str) -> bool:
    """Return whether a character may anchor a non-dangling ZWJ."""
    if ch in {ZWJ, VS16, VS15, KEYCAP}:
        return False
    if _is_tag_char(ch):
        return False
    return True


def _collapse_zwj(text: str) -> str:
    """Collapse repeated zero-width joiners introduced by stripping passes."""
    collapsed = text
    while ZWJ * 2 in collapsed:
        collapsed = collapsed.replace(ZWJ * 2, ZWJ)
    return collapsed


def _remove_dangling_zwj(text: str) -> str:
    """Keep only ZWJ characters connecting two base characters."""
    if not text:
        return text
    chars = list(text)

    # A joiner is retained only when it has usable bases on both sides.
    def is_base(index: int) -> bool:
        return 0 <= index < len(chars) and _is_base_for_zwj(chars[index])

    out: list[str] = []
    for i, ch in enumerate(chars):
        if ch == ZWJ:
            if is_base(i - 1) and is_base(i + 1):
                out.append(ch)
            continue
        out.append(ch)
    return "".join(out)


def normalize_emoji_token(
    token: str,
    *,
    strip_variation_selectors: bool = True,
    strip_skin_tone: bool = True,
    strip_gender: bool = False,
) -> str:
    """Normalize one emoji segment according to configured WNS rules.

    The default settings match the JADT camera-ready analysis: variation
    selectors and skin-tone modifiers are stripped, while gendered emoji
    sequences are retained as distinct types.
    """
    normalized = str(token)

    # Preserve the pass order from the WNS utility because stripping can create
    # dangling ZWJ characters that must then be cleaned consistently.
    if strip_variation_selectors:
        normalized = _strip_variation_selectors(normalized)
    if strip_skin_tone:
        normalized = _strip_skin_tone(normalized)
    if strip_gender:
        normalized = _strip_gender(normalized)

    # Leading/trailing ZWJ can arise from segmentation around gender markers.
    normalized = normalized.strip(ZWJ)
    if strip_skin_tone or strip_gender:
        normalized = _collapse_zwj(normalized)
        normalized = _remove_dangling_zwj(normalized)
        normalized = normalized.strip(ZWJ)
        normalized = _collapse_zwj(normalized)
        normalized = normalized.strip(ZWJ)
    return normalized


def _is_emoji_token(token: object) -> bool:
    """Return whether an ``emoji.analyze`` token is an emoji token."""
    return isinstance(getattr(token, "value", None), EmojiMatch)


def _merge_zwj_sequences(tokens: list[str]) -> list[str]:
    """Merge adjacent emoji tokens connected by explicit ZWJ tokens."""
    if not tokens:
        return tokens

    # ``emoji.analyze(..., join_emoji=False)`` can yield ZWJ as its own token
    # for non-RGI sequences.  The WNS utility re-joins those pieces here.
    merged: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == ZWJ:
            i += 1
            continue
        while i + 2 < len(tokens) and tokens[i + 1] == ZWJ and tokens[i + 2] != ZWJ:
            token = token + ZWJ + tokens[i + 2]
            i += 2
        merged.append(token)
        i += 1
    return merged


def extract_runs(
    text: object,
    *,
    mode: str,
    join_emoji_sequences: bool,
    delimiter: str = "-",
    strip_variation_selectors: bool = True,
    strip_skin_tone: bool = True,
    strip_gender: bool = False,
) -> list[str]:
    """Extract maximal emoji or text runs using the WNS utility algorithm.

    In ``emoji`` mode, adjacent emoji segments form a run and are joined by
    ``delimiter``.  In ``text`` mode, maximal non-emoji runs between emoji runs
    are returned unchanged except that whitespace-only runs are dropped.
    """
    if text is None:
        return []
    source = str(text)
    if not source:
        return []

    mode_normalized = str(mode).strip().lower()
    if mode_normalized not in {"emoji", "text"}:
        raise ValueError("mode must be 'emoji' or 'text'")
    if not delimiter:
        raise ValueError("delimiter cannot be empty")

    out: list[str] = []
    current_text: list[str] = []
    raw_emoji_tokens: list[str] = []
    want_emoji = mode_normalized == "emoji"

    def flush_current() -> None:
        """Flush the current run according to the selected extraction mode."""
        nonlocal current_text, raw_emoji_tokens
        if want_emoji:
            segments = _merge_zwj_sequences(raw_emoji_tokens) if join_emoji_sequences else raw_emoji_tokens
            normalized_segments: list[str] = []
            for segment in segments:
                normalized = normalize_emoji_token(
                    segment,
                    strip_variation_selectors=strip_variation_selectors,
                    strip_skin_tone=strip_skin_tone,
                    strip_gender=strip_gender,
                )
                if normalized:
                    normalized_segments.append(normalized)
            if normalized_segments:
                out.append(delimiter.join(normalized_segments))
            raw_emoji_tokens = []
        else:
            if current_text:
                run = "".join(current_text)
                if run.strip():
                    out.append(run)
        current_text = []

    # Keep the WNS utility contract: always request unjoined analysis, then
    # handle ZWJ joining ourselves so run boundaries remain observable.
    for token in emoji.analyze(source, join_emoji=False):
        chars = getattr(token, "chars", "")
        token_is_emoji = _is_emoji_token(token)

        if token_is_emoji == want_emoji:
            if want_emoji:
                raw_emoji_tokens.append(chars)
            elif chars:
                current_text.append(chars)
            continue

        # ZWJ between emoji pieces belongs to the emoji buffer, not to text.
        if want_emoji:
            if chars == ZWJ:
                raw_emoji_tokens.append(chars)
            else:
                flush_current()
        else:
            flush_current()

    flush_current()
    return out


def extract_emoji_runs(
    text: object,
    *,
    delimiter: str = "-",
    join_emoji_sequences: bool = True,
    strip_variation_selectors: bool = True,
    strip_skin_tone: bool = True,
    strip_gender: bool = False,
) -> list[str]:
    """Extract delimiter-coded contiguous emoji runs from message text."""
    return extract_runs(
        text,
        mode="emoji",
        join_emoji_sequences=join_emoji_sequences,
        delimiter=delimiter,
        strip_variation_selectors=strip_variation_selectors,
        strip_skin_tone=strip_skin_tone,
        strip_gender=strip_gender,
    )


def extract_text_runs(
    text: object,
    *,
    join_emoji_sequences: bool = True,
    delimiter: str = "-",
) -> list[str]:
    """Extract maximal non-emoji text runs from message text.

    The delimiter is accepted for API symmetry with emoji extraction but is not
    used to join text tokens.  It is still validated by ``extract_runs``.
    """
    return extract_runs(
        text,
        mode="text",
        join_emoji_sequences=join_emoji_sequences,
        delimiter=delimiter,
        strip_variation_selectors=False,
        strip_skin_tone=False,
        strip_gender=False,
    )


def _is_system_source(source_id: str, generated_by: str) -> bool:
    """Return whether a post belongs to a WNS system pseudo-source."""
    return bool(SYSTEM_SOURCE_RE.match(str(source_id).strip())) or str(generated_by).strip().lower() == "system"


def read_posts_tsv(path: str | Path, *, sep: str = "\t") -> list[dict[str, str]]:
    """Read the minimal post-level TSV into row dictionaries."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=sep)
        if reader.fieldnames is None:
            raise ValueError(f"Post TSV has no header row: {path}")
        return [dict(row) for row in reader]


def write_emoji_tsv(
    posts_path: str | Path,
    output_path: str | Path,
    *,
    source_col: str = "source_id",
    text_col: str = "text",
    delimiter: str = "-",
    sep: str = "\t",
    include_system: bool = False,
    join_emoji_sequences: bool = True,
    strip_variation_selectors: bool = True,
    strip_skin_tone: bool = True,
    strip_gender: bool = False,
) -> None:
    """Write the two-column WNS-derived emoji TSV used by table scripts."""
    rows = read_posts_tsv(posts_path, sep=sep)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # The public reproduction input contains only source identifiers and runs.
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "sequence"], delimiter=sep, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if not include_system and _is_system_source(row.get(source_col, ""), row.get("generated_by", "")):
                continue
            for run in extract_emoji_runs(
                row.get(text_col, ""),
                delimiter=delimiter,
                join_emoji_sequences=join_emoji_sequences,
                strip_variation_selectors=strip_variation_selectors,
                strip_skin_tone=strip_skin_tone,
                strip_gender=strip_gender,
            ):
                writer.writerow({"source_id": row.get(source_col, ""), "sequence": run})


def write_lexical_tsv(
    posts_path: str | Path,
    output_path: str | Path,
    *,
    source_col: str = "source_id",
    text_col: str = "text",
    sep: str = "\t",
    include_system: bool = False,
    join_emoji_sequences: bool = True,
) -> None:
    """Write the two-column WNS-derived text-run TSV used by table scripts.

    This mirrors the original WNS ``mode=text`` preparation step: each maximal
    non-emoji run becomes one output row, and whitespace-only runs are omitted.
    The downstream lexical table script still owns lowercasing, word-token
    extraction, and placeholder/artifact filtering.
    """
    rows = read_posts_tsv(posts_path, sep=sep)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Emit text runs rather than full posts to match the WNS no-emoji TSV input.
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "text"], delimiter=sep, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if not include_system and _is_system_source(row.get(source_col, ""), row.get("generated_by", "")):
                continue
            for run in extract_text_runs(row.get(text_col, ""), join_emoji_sequences=join_emoji_sequences):
                writer.writerow({"source_id": row.get(source_col, ""), "text": run})
