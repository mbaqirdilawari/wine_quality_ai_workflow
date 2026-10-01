"""Paths read from environment variables, with defaults.

Variables are read inside the functions (not at import time) so tests can
override them with ``monkeypatch``.
"""

import os
from pathlib import Path

DEFAULT_DATA_PATH = "data/wine_quality_merged.csv"
DEFAULT_OUTPUT_DIR = "outputs"


def get_data_path() -> Path:
    """Return the input CSV path from ``WINE_DATA_PATH`` or the default."""
    return Path(os.environ.get("WINE_DATA_PATH") or DEFAULT_DATA_PATH)


def get_output_dir() -> Path:
    """Return the chart output dir from ``WINE_OUTPUT_DIR``, creating it if missing."""
    out_dir = Path(os.environ.get("WINE_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
