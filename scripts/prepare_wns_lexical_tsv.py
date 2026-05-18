#!/usr/bin/env python
"""Prepare the JADT lexical TSV from a local WNS post-level TSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from typical_source_estimation.wns import write_lexical_tsv


def main() -> None:
    """Run the WNS posts-to-lexical preparation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posts", required=True, help="Input post-level TSV path")
    parser.add_argument("--output", required=True, help="Output two-column lexical TSV path")
    parser.add_argument("--include-system", action="store_true", help="Keep WNS system pseudo-source rows")
    args = parser.parse_args()

    # The lexical table script performs tokenization and placeholder filtering.
    write_lexical_tsv(
        Path(args.posts),
        Path(args.output),
        include_system=bool(args.include_system),
    )
    print(f"Wrote WNS lexical TSV: {args.output}")


if __name__ == "__main__":
    main()
