"""Tests for WNS preparation helpers and tiny XML fixtures."""

from __future__ import annotations

import csv
from pathlib import Path

from typical_source_estimation.wns import (
    extract_emoji_runs,
    iter_wns_posts,
    normalize_emoji_token,
    prepare_posts_tsv,
    write_emoji_tsv,
    write_lexical_tsv,
)


FIXTURE_XML_DIR = Path(__file__).parent / "fixtures" / "wns_xml"


def _read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a small TSV fixture output into dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_iter_wns_posts_extracts_minimal_post_records() -> None:
    """WNS XML parsing keeps post metadata and excludes timestamp text."""
    posts = list(iter_wns_posts(FIXTURE_XML_DIR))

    # The fixture contains three human posts and one system post.
    assert len(posts) == 4
    assert posts[0].post_id == "wns.chat.01.1"
    assert posts[0].source_id == "#wns.user.001"
    assert posts[0].generated_by == "human"
    assert posts[0].text == "Salut 😂😂 ok 🤷🏻\u200d♀️!"


def test_emoji_normalization_strips_skin_and_variation_but_keeps_gender() -> None:
    """The default paper contract removes skin tone/VS and retains gender."""
    normalized = normalize_emoji_token("🤷🏻\u200d♀️")

    # Skin-tone and VS codepoints are gone, but the gender sign remains.
    assert normalized == "🤷\u200d♀"
    assert "🏻" not in normalized
    assert "️" not in normalized


def test_extract_emoji_runs_preserves_contiguous_runs() -> None:
    """Emoji extraction emits delimiter-coded runs separated by text."""
    runs = extract_emoji_runs("Hi 😱😱 ok 🤷🏻\u200d♀️😊")

    # Adjacent emoji stay in the same run; intervening text starts a new run.
    assert runs == ["😱-😱", "🤷\u200d♀-😊"]


def test_prepare_posts_tsv_writes_minimal_columns(tmp_path: Path) -> None:
    """The post-level script output is a minimal TSV for downstream steps."""
    out_path = tmp_path / "posts.tsv"
    prepare_posts_tsv(FIXTURE_XML_DIR, out_path)
    rows = _read_tsv(out_path)

    # The output keeps only fields needed to regenerate analysis inputs.
    assert list(rows[0]) == ["source_file", "post_id", "source_id", "generated_by", "modality", "text"]
    assert rows[0]["source_id"] == "#wns.user.001"
    assert rows[0]["text"].startswith("Salut 😂😂")


def test_write_emoji_tsv_excludes_system_and_outputs_two_columns(tmp_path: Path) -> None:
    """Emoji TSV preparation writes only source IDs and emoji sequences."""
    posts_path = tmp_path / "posts.tsv"
    emoji_path = tmp_path / "emoji.tsv"
    prepare_posts_tsv(FIXTURE_XML_DIR, posts_path)
    write_emoji_tsv(posts_path, emoji_path)
    rows = _read_tsv(emoji_path)

    # The first post has two emoji runs; system rows do not appear.
    assert list(rows[0]) == ["source_id", "sequence"]
    assert rows == [
        {"source_id": "#wns.user.001", "sequence": "😂-😂"},
        {"source_id": "#wns.user.001", "sequence": "🤷\u200d♀"},
    ]


def test_write_lexical_tsv_excludes_system_and_preserves_text(tmp_path: Path) -> None:
    """Lexical TSV preparation leaves tokenization to later table scripts."""
    posts_path = tmp_path / "posts.tsv"
    lexical_path = tmp_path / "lexical.tsv"
    prepare_posts_tsv(FIXTURE_XML_DIR, posts_path)
    write_lexical_tsv(posts_path, lexical_path)
    rows = _read_tsv(lexical_path)

    # Human media placeholders remain available for later documented filtering.
    assert list(rows[0]) == ["source_id", "text"]
    assert [row["source_id"] for row in rows] == ["#wns.user.001", "#wns.user.002", "#wns.user.001"]
    assert rows[1]["text"] == "_MEDIA_OMITTED_"
