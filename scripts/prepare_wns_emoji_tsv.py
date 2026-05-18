#!/usr/bin/env python
"""Prepare the JADT emoji TSV from a local WNS post-level TSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from typical_source_estimation.wns import write_emoji_tsv


def main() -> None:
    """Run the WNS posts-to-emoji preparation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posts", required=True, help="Input post-level TSV path")
    parser.add_argument("--output", required=True, help="Output two-column emoji TSV path")
    parser.add_argument("--delimiter", default="-", help="Delimiter between emoji tokens in a run")
    parser.add_argument("--include-system", action="store_true", help="Keep WNS system pseudo-source rows")
    parser.add_argument("--keep-skin-tone", action="store_true", help="Do not strip skin-tone modifiers")
    parser.add_argument("--keep-variation-selectors", action="store_true", help="Do not strip variation selectors")
    parser.add_argument("--strip-gender", action="store_true", help="Strip gender signs from emoji sequences")
    args = parser.parse_args()

    # Defaults match the camera-ready paper: strip VS and skin tone, keep gender.
    write_emoji_tsv(
        Path(args.posts),
        Path(args.output),
        delimiter=str(args.delimiter),
        include_system=bool(args.include_system),
        strip_variation_selectors=not bool(args.keep_variation_selectors),
        strip_skin_tone=not bool(args.keep_skin_tone),
        strip_gender=bool(args.strip_gender),
    )
    print(f"Wrote WNS emoji TSV: {args.output}")


if __name__ == "__main__":
    main()
