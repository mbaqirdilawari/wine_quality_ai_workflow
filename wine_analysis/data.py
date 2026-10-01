"""Load, inspect and clean the wine quality dataset."""

from pathlib import Path

import pandas as pd

CHEMICAL_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
TARGET_COLUMN = "quality"
TYPE_COLUMN = "type"
EXPECTED_COLUMNS = CHEMICAL_COLUMNS + [TARGET_COLUMN, TYPE_COLUMN]


def load_data(path) -> pd.DataFrame:
    """Read the CSV at ``path`` and check that all expected columns are present."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Data file not found: {path}. Set WINE_DATA_PATH to the CSV location."
        )
    df = pd.read_csv(path)
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Data file {path} is missing expected columns: {missing}")
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """Return the shape, missing values per column and exact duplicate count."""
    return {
        "shape": df.shape,
        "missing": {col: int(n) for col, n in df.isna().sum().items()},
        "duplicates": int(df.duplicated().sum()),
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with exact duplicates and rows with missing values dropped.

    Duplicates are dropped first, then incomplete rows; the index is reset.
    Missing values are not imputed, and outliers are kept on purpose: both are
    real lab-measured wines, and imputing would invent measurements.
    """
    return df.drop_duplicates().dropna().reset_index(drop=True)
