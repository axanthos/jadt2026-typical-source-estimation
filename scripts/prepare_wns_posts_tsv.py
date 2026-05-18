#!/usr/bin/env python
"""Prepare a minimal post-level TSV from controlled-access WNS XML-TEI files."""

from __future__ import annotations

import argparse
from pathlib import Path

from typical_source_estimation.wns import prepare_posts_tsv


def main() -> None:
    """Run the WNS XML-to-posts preparation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-dir", required=True, help="Directory containing WNS XML-TEI files")
    parser.add_argument("--output", required=True, help="Output post-level TSV path")
    parser.add_argument("--pattern", default="wns_chat_*.xml", help="Glob pattern for XML files")
    args = parser.parse_args()

    # Keep the script thin so the tested package module owns the behavior.
    prepare_posts_tsv(Path(args.xml_dir), Path(args.output), pattern=str(args.pattern))
    print(f"Wrote WNS post TSV: {args.output}")


if __name__ == "__main__":
    main()
