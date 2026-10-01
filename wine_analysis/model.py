"""Linear regression of quality on chemical features."""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from wine_analysis.data import CHEMICAL_COLUMNS, TARGET_COLUMN

FEATURES_BASIC = ["alcohol", "volatile acidity", "sulphates"]
FEATURES_ALL = list(CHEMICAL_COLUMNS)  # all 11 chemical features, `type` excluded

# Below this the 20% test split is too small for a meaningful R².
MIN_ROWS = 10


def train_and_evaluate(
    df: pd.DataFrame, features, test_size: float = 0.2, random_state: int = 42
) -> dict:
    """Fit a linear regression on a train split and score it on the test split."""
    features = list(features)
    unknown = [f for f in features if f not in df.columns]
    if not features or unknown:
        raise ValueError(f"Unknown or empty feature list: {unknown or features}")
    if len(df) < MIN_ROWS:
        raise ValueError(f"Need at least {MIN_ROWS} rows to train, got {len(df)}")

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[TARGET_COLUMN], test_size=test_size, random_state=random_state
    )
    model = LinearRegression().fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "features": features,
        "r2": float(r2_score(y_test, predictions)),
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
