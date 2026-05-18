"""Preparation helpers for controlled-access WNS real-data reproduction.

The public reproduction package does not redistribute token-level material from
*What's New, Switzerland?* (WNS).  Authorized users can run the helpers in this
module against their local controlled-access copy of the XML-TEI corpus to
produce the small TSV inputs consumed by the paper's real-data scripts.
"""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


XML_NS = "http://www.w3.org/XML/1998/namespace"
XML_ID = f"{{{XML_NS}}}id"
ZWJ = "\u200d"
VS15 = "\ufe0e"
VS16 = "\ufe0f"
KEYCAP = "\u20e3"
GENDER_SIGNS = {"\u2640", "\u2642"}
SKIN_TONE_MIN = 0x1F3FB
SKIN_TONE_MAX = 0x1F3FF
SYSTEM_SOURCE_RE = re.compile(r"(?i)^(#?wns\.system|#?system|system)$")


@dataclass(frozen=True, slots=True)
class WnsPost:
    """Minimal post record extracted from a WNS XML-TEI file.

    The fields are deliberately limited to what is needed to regenerate the
    paper's emoji and lexical TSV inputs.  Richer WNS metadata remain available
    through the controlled-access corpus itself rather than this reproduction
    package.
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

        # Child tails belong to the surrounding post text even for <time>.
        for child in list(element):
            if _local_name(child.tag) != "time":
                collect(child)
            if child.tail:
                pieces.append(child.tail)

    # Strip outer whitespace introduced by pretty-printed XML documents.
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

    # Use the standard CSV writer to keep quoting/escaping explicit.
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


def _is_variation_selector(ch: str) -> bool:
    """Return whether ``ch`` is an emoji/text presentation selector."""
    return ch in {VS15, VS16}


def _is_skin_tone(ch: str) -> bool:
    """Return whether ``ch`` is a Fitzpatrick skin-tone modifier."""
    return SKIN_TONE_MIN <= ord(ch) <= SKIN_TONE_MAX


def _is_regional_indicator(ch: str) -> bool:
    """Return whether ``ch`` is a regional-indicator flag component."""
    return 0x1F1E6 <= ord(ch) <= 0x1F1FF


def _is_emoji_base(ch: str) -> bool:
    """Approximate whether a Unicode character can start an emoji token.

    The ranges cover the emoji blocks encountered in the WNS-derived paper
    tables and common symbol emoji.  The function is intentionally local and
    dependency-free; it is not a complete replacement for the Unicode emoji
    specification.
    """
    code = ord(ch)

    # Cover pictographs, dingbats, miscellaneous symbols, arrows, and clocks.
    return (
        0x1F000 <= code <= 0x1FAFF
        or 0x2600 <= code <= 0x27BF
        or 0x2300 <= code <= 0x23FF
        or 0x2B00 <= code <= 0x2BFF
    )


def _consume_modifiers(text: str, index: int) -> int:
    """Advance over optional variation selectors and skin-tone modifiers."""
    i = int(index)
    while i < len(text) and (_is_variation_selector(text[i]) or _is_skin_tone(text[i])):
        i += 1
    return i


def _read_keycap_token(text: str, index: int) -> tuple[str | None, int]:
    """Read a keycap emoji sequence when one starts at ``index``."""
    ch = text[index]
    if ch not in set("0123456789#*"):
        return None, index

    # A keycap may be encoded as digit + VS16 + U+20E3 or digit + U+20E3.
    i = index + 1
    if i < len(text) and _is_variation_selector(text[i]):
        i += 1
    if i < len(text) and text[i] == KEYCAP:
        return text[index : i + 1], i + 1
    return None, index


def _read_emoji_token(text: str, index: int) -> tuple[str | None, int]:
    """Read one emoji token or ZWJ sequence from ``text`` at ``index``."""
    keycap, next_index = _read_keycap_token(text, index)
    if keycap is not None:
        return keycap, next_index

    # Non-emoji characters are boundaries between emoji runs.
    if index >= len(text) or not _is_emoji_base(text[index]):
        return None, index

    start = index
    i = _consume_modifiers(text, index + 1)

    # Regional indicators form flag pairs when two appear consecutively.
    if _is_regional_indicator(text[start]) and i < len(text) and _is_regional_indicator(text[i]):
        i = _consume_modifiers(text, i + 1)

    # ZWJ chains are kept as a single token, including gendered emoji sequences.
    while i + 1 < len(text) and text[i] == ZWJ and _is_emoji_base(text[i + 1]):
        i = _consume_modifiers(text, i + 2)
    return text[start:i], i


def _cleanup_zwj(token: str) -> str:
    """Remove leading/trailing or repeated joiners introduced by stripping."""
    cleaned = token
    while ZWJ * 2 in cleaned:
        cleaned = cleaned.replace(ZWJ * 2, ZWJ)
    return cleaned.strip(ZWJ)


def normalize_emoji_token(
    token: str,
    *,
    strip_variation_selectors: bool = True,
    strip_skin_tone: bool = True,
    strip_gender: bool = False,
) -> str:
    """Normalize one emoji token according to the JADT preprocessing contract."""
    normalized = str(token)

    # The paper removes presentation variants and skin tone, but keeps gender.
    if strip_variation_selectors:
        normalized = "".join(ch for ch in normalized if not _is_variation_selector(ch))
    if strip_skin_tone:
        normalized = "".join(ch for ch in normalized if not _is_skin_tone(ch))

    # Gender stripping is optional for diagnostic reuse, not the paper default.
    if strip_gender:
        normalized = "".join(ch for ch in normalized if ch not in GENDER_SIGNS)
    return _cleanup_zwj(normalized)


def extract_emoji_runs(
    text: object,
    *,
    delimiter: str = "-",
    strip_variation_selectors: bool = True,
    strip_skin_tone: bool = True,
    strip_gender: bool = False,
) -> list[str]:
    """Extract delimiter-coded contiguous emoji runs from message text."""
    if text is None:
        return []
    source = str(text)
    if not source:
        return []

    runs: list[str] = []
    current_tokens: list[str] = []
    i = 0

    # Scan left-to-right, flushing the current emoji run on text boundaries.
    while i < len(source):
        token, next_i = _read_emoji_token(source, i)
        if token is None:
            if current_tokens:
                runs.append(delimiter.join(current_tokens))
                current_tokens = []
            i += 1
            continue

        # Normalize each atomic emoji or ZWJ sequence before adding it to a run.
        normalized = normalize_emoji_token(
            token,
            strip_variation_selectors=strip_variation_selectors,
            strip_skin_tone=strip_skin_tone,
            strip_gender=strip_gender,
        )
        if normalized:
            current_tokens.append(normalized)
        i = next_i

    # Emit the final run when the message ends with emoji.
    if current_tokens:
        runs.append(delimiter.join(current_tokens))
    return runs


def _is_system_source(source_id: str, generated_by: str) -> bool:
    """Return whether a post belongs to a system pseudo-source."""
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
) -> None:
    """Write the two-column WNS-derived lexical TSV used by table scripts."""
    rows = read_posts_tsv(posts_path, sep=sep)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Lexical tokenization/filtering happens later in the final table script.
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "text"], delimiter=sep, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if not include_system and _is_system_source(row.get(source_col, ""), row.get("generated_by", "")):
                continue
            text = str(row.get(text_col, "") or "")
            if text.strip():
                writer.writerow({"source_id": row.get(source_col, ""), "text": text})
