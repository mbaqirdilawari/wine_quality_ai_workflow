"""Run the full pipeline: load -> inspect -> clean -> explore -> model -> charts.

This is the only module that reads config and prints. Run it with
``python -m wine_analysis.main``.
"""

import pandas as pd

from wine_analysis.config import get_data_path, get_output_dir
from wine_analysis.data import clean_data, inspect_data, load_data
from wine_analysis.explore import (
    filter_by_type,
    filter_min_alcohol,
    filter_min_quality,
    summary_by_quality,
    summary_by_type,
)
from wine_analysis.model import FEATURES_ALL, FEATURES_BASIC, train_and_evaluate
from wine_analysis.plots import plot_alcohol_by_quality, plot_alcohol_vs_density

DUPLICATES_REASON = (
    "Identical rows over-weight some wines and can land in both the train and "
    "test splits, which inflates test scores."
)
MISSING_REASON = (
    "The model and the trend line cannot handle missing values. Values are not "
    "imputed, because that would invent lab measurements."
)
OUTLIERS_REASON = (
    "Outliers are real, lab-measured wines, not entry errors. Removing them would "
    "bias the model toward 'average' wines and drop most rare quality-3 and "
    "quality-9 wines."
)


def _heading(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def _table(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda v: f"{v:.3f}")


def main() -> None:
    data_path = get_data_path()
    out_dir = get_output_dir()
    print(f"Data:    {data_path}\nOutputs: {out_dir}")

    raw = load_data(data_path)

    _heading("1. INSPECT (raw data)")
    info = inspect_data(raw)
    print(f"Shape: {info['shape']}")
    print(f"Missing values (total): {sum(info['missing'].values())}")
    for col, n in info["missing"].items():
        print(f"  {col:<22} {n}")
    print(f"Exact duplicate rows: {info['duplicates']}")

    _heading("2. CLEANING")
    df = clean_data(raw)
    # clean_data drops duplicates first, so the rest of the drop is missing values.
    n_missing_rows = len(raw) - info["duplicates"] - len(df)
    print(f"Removed {info['duplicates']} exact duplicate rows")
    print(f"Removed {n_missing_rows} rows with missing values")
    print(f"-> {len(df)} rows remain")
    print(f"Rows by type: {df['type'].value_counts().to_dict()}")
    print(f"Why remove duplicates: {DUPLICATES_REASON}")
    print(f"Why remove rows with missing values: {MISSING_REASON}")
    print(f"Outliers kept: {OUTLIERS_REASON}")

    _heading("3. EXPLORATION")
    print(f"Wines with alcohol >= 12%: {len(filter_min_alcohol(df, 12.0))}")
    print(f"Red wines only:            {len(filter_by_type(df, 'red'))}")
    print(f"High quality (>= 7):       {len(filter_min_quality(df, 7))}")
    print("\nMean of key features by type:")
    print(_table(summary_by_type(df)))
    print("\nCount and mean alcohol by quality:")
    print(_table(summary_by_quality(df)))

    _heading("4. MODEL COMPARISON (linear regression, 80/20 split)")
    results = {
        "3 features": train_and_evaluate(df, FEATURES_BASIC),
        "11 features": train_and_evaluate(df, FEATURES_ALL),
    }
    print(f"{'Feature set':<14}{'R²':>8}{'RMSE':>8}{'Train':>8}{'Test':>8}")
    for name, r in results.items():
        print(
            f"{name:<14}{r['r2']:>8.3f}{r['rmse']:>8.3f}"
            f"{r['n_train']:>8}{r['n_test']:>8}"
        )
    basic, full = results["3 features"], results["11 features"]
    print(
        f"\nUsing all 11 features changes R² by {full['r2'] - basic['r2']:+.3f} "
        f"and RMSE by {full['rmse'] - basic['rmse']:+.3f}. "
        f"3 features ({', '.join(FEATURES_BASIC)}) explain about "
        f"{basic['r2']:.0%} of the variance in quality; all 11 explain about "
        f"{full['r2']:.0%}."
    )

    _heading("5. CHARTS")
    for path in (
        plot_alcohol_by_quality(df, out_dir),
        plot_alcohol_vs_density(df, out_dir),
    ):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
