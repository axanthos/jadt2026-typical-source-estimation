"""Tests for WNS preparation helpers and tiny XML fixtures."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from typical_source_estimation.wns import (
    _merge_zwj_sequences,
    extract_emoji_runs,
    extract_runs,
    extract_text_runs,
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


def test_normalize_emoji_token_matches_jadt_defaults() -> None:
    """The default paper contract removes skin tone/VS and retains gender."""
    normalized = normalize_emoji_token("🤷🏻\u200d♀️")

    # Skin-tone and VS codepoints are gone, but the gender sign remains.
    assert normalized == "🤷\u200d♀"
    assert "🏻" not in normalized
    assert "️" not in normalized


def test_normalize_emoji_token_can_reproduce_original_gender_stripping() -> None:
    """The port preserves the original WNS utility's gender-stripping option."""
    normalized = normalize_emoji_token(
        "🙇🏻\u200d♀️",
        strip_variation_selectors=True,
        strip_skin_tone=False,
        strip_gender=True,
    )

    # Gender is removed and the now-dangling ZWJ is cleaned, but skin tone stays.
    assert normalized == "🙇🏻"
    assert "♀" not in normalized
    assert "\u200d" not in normalized


def test_extract_emoji_runs_uses_emoji_package_segmentation() -> None:
    """Emoji extraction emits delimiter-coded runs separated by non-emoji text."""
    runs = extract_emoji_runs("Hi 😱😱 ok 🤷🏻\u200d♀️😊")

    # Adjacent emoji stay in the same run; intervening text starts a new run.
    assert runs == ["😱-😱", "🤷\u200d♀-😊"]


def test_variation_selectors_and_skin_tone_match_original_options() -> None:
    """Configurable normalization reproduces the WNS utility edge cases."""
    assert extract_emoji_runs("❤❤️", strip_variation_selectors=False, strip_skin_tone=False) == ["❤-❤️"]
    assert extract_emoji_runs("❤❤️", strip_variation_selectors=True, strip_skin_tone=False) == ["❤-❤"]
    assert extract_emoji_runs("👍🏾👍", strip_skin_tone=True) == ["👍-👍"]


def test_zwj_sequence_merging_matches_original_helper() -> None:
    """The direct ZWJ merge helper preserves the original utility contract."""
    family = "👨\u200d👩\u200d👧"

    # Explicit ZWJ connector tokens are merged, while bare joiners are dropped.
    assert _merge_zwj_sequences(["👨", "\u200d", "👩", "\u200d", "👧"]) == [family]
    assert _merge_zwj_sequences(["🙋", "\u200d"]) == ["🙋"]
    assert _merge_zwj_sequences(["\u200d", "🙋"]) == ["🙋"]


def test_text_mode_extracts_non_emoji_runs() -> None:
    """Text extraction mirrors the WNS utility's ``mode=text`` behavior."""
    assert extract_text_runs("Hello 😂 world 😊 bye") == ["Hello ", " world ", " bye"]
    assert extract_text_runs("😂   😊") == []
    assert extract_runs("😂😊👍", mode="text", join_emoji_sequences=True) == []


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


def test_write_lexical_tsv_excludes_system_and_writes_text_runs(tmp_path: Path) -> None:
    """Lexical TSV preparation writes WNS ``mode=text`` runs, not full posts."""
    posts_path = tmp_path / "posts.tsv"
    lexical_path = tmp_path / "lexical.tsv"
    prepare_posts_tsv(FIXTURE_XML_DIR, posts_path)
    write_lexical_tsv(posts_path, lexical_path)
    rows = _read_tsv(lexical_path)

    # Emoji tokens are removed, but each non-emoji run is retained exactly.
    assert list(rows[0]) == ["source_id", "text"]
    assert rows == [
        {"source_id": "#wns.user.001", "text": "Salut "},
        {"source_id": "#wns.user.001", "text": " ok "},
        {"source_id": "#wns.user.001", "text": "!"},
        {"source_id": "#wns.user.002", "text": "_MEDIA_OMITTED_"},
        {"source_id": "#wns.user.001", "text": "Texte sans emoji."},
    ]


def test_invalid_extraction_mode_is_rejected() -> None:
    """WNS run extraction fails early for unsupported modes."""
    with pytest.raises(ValueError, match="mode"):
        extract_runs("abc", mode="invalid", join_emoji_sequences=True)
