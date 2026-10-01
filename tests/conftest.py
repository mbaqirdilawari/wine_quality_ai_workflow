"""Small synthetic DataFrame fixtures shared by the unit tests."""

import numpy as np
import pandas as pd
import pytest

from wine_analysis.data import CHEMICAL_COLUMNS, EXPECTED_COLUMNS


def _base_row(**overrides):
    row = {
        "fixed acidity": 7.0,
        "volatile acidity": 0.3,
        "citric acid": 0.3,
        "residual sugar": 2.0,
        "chlorides": 0.05,
        "free sulfur dioxide": 30.0,
        "total sulfur dioxide": 120.0,
        "density": 0.995,
        "pH": 3.2,
        "sulphates": 0.5,
        "alcohol": 10.0,
        "quality": 6,
        "type": "white",
    }
    row.update(overrides)
    return row


@pytest.fixture
def wine_df():
    """Six distinct wines (3 red, 3 white) with known values for filters/summaries."""
    rows = [
        _base_row(alcohol=9.0, quality=5, type="red", density=0.998),
        _base_row(alcohol=10.0, quality=5, type="red", density=0.997),
        _base_row(alcohol=12.5, quality=7, type="red", density=0.994),
        _base_row(alcohol=11.0, quality=6, type="white", density=0.995),
        _base_row(alcohol=12.0, quality=7, type="white", density=0.993),
        _base_row(alcohol=13.0, quality=8, type="white", density=0.991),
    ]
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


@pytest.fixture
def dup_df():
    """Four rows: one exact duplicate pair, one near-duplicate, one NaN."""
    rows = [
        _base_row(),
        _base_row(),  # exact duplicate of row 0
        _base_row(alcohol=10.1),  # differs in a single value
        _base_row(chlorides=np.nan),
    ]
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


@pytest.fixture
def linear_df():
    """Quality is an exact linear function of the chemical features."""
    rng = np.random.default_rng(0)
    n = 60
    data = {col: rng.uniform(0.1, 10.0, n) for col in CHEMICAL_COLUMNS}
    df = pd.DataFrame(data)
    df["quality"] = 2.0 + 0.5 * df["alcohol"] - 1.5 * df["volatile acidity"]
    df["quality"] += 0.8 * df["sulphates"]
    df["type"] = np.where(np.arange(n) % 2 == 0, "red", "white")
    return df
