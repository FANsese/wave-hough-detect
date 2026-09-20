"""
pytest configuration.

The repository uses a src/ layout and the tests run directly against the source
tree (no ``pip install -e .`` needed first), so src/ is added to the import path
here.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
