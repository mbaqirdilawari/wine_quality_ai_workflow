"""Light exploration: row filters and group summaries."""

import pandas as pd

KEY_FEATURES = ["alcohol", "volatile acidity", "sulphates", "density", "quality"]


def filter_min_alcohol(df: pd.DataFrame, threshold: float = 12.0) -> pd.DataFrame:
    """Return wines with alcohol >= ``threshold``."""
    return df[df["alcohol"] >= threshold]


def filter_by_type(df: pd.DataFrame, wine_type: str = "red") -> pd.DataFrame:
    """Return wines of a single ``type`` (``red`` or ``white``)."""
    return df[df["type"] == wine_type]


def filter_min_quality(df: pd.DataFrame, threshold: int = 7) -> pd.DataFrame:
    """Return wines with quality >= ``threshold``."""
    return df[df["quality"] >= threshold]


def summary_by_type(df: pd.DataFrame, features=None) -> pd.DataFrame:
    """Return the mean of key features for each wine type."""
    features = KEY_FEATURES if features is None else features
    return df.groupby("type")[features].mean()


def summary_by_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Return the wine count and mean alcohol for each quality score."""
    return df.groupby("quality").agg(
        count=("alcohol", "size"), mean_alcohol=("alcohol", "mean")
    )
