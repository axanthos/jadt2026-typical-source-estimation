"""Pytest configuration for source-tree imports during local development."""

from __future__ import annotations

import sys
from pathlib import Path


# Allow `python -m pytest` from a fresh checkout before editable installation.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
