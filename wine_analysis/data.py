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
NUMERIC_COLUMNS = CHEMICAL_COLUMNS + [TARGET_COLUMN]


def load_data(path) -> pd.DataFrame:
    """Read the CSV at ``path`` and return only the 13 expected columns.

    Extra columns are dropped so they cannot affect duplicate detection or the
    missing-value check. Raises ``ValueError`` if an expected column is missing,
    the file has no data rows, or a numeric column holds non-numeric values.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Data file not found: {path}. Set WINE_DATA_PATH to the CSV location."
        )
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Data file {path} is empty") from None
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Data file {path} is missing expected columns: {missing}")
    if df.empty:
        raise ValueError(f"Data file {path} has no data rows")
    df = df[EXPECTED_COLUMNS]
    non_numeric = [
        col for col in NUMERIC_COLUMNS if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise ValueError(
            f"Data file {path} has non-numeric values in columns: {non_numeric}"
        )
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
