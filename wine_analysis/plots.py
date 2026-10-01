"""Save the two analysis charts as PNG files."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed locally or in Docker

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BOXPLOT_FILENAME = "alcohol_by_quality_boxplot.png"
SCATTER_FILENAME = "alcohol_vs_density_scatter.png"


def _check_not_empty(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Cannot plot an empty DataFrame")


def _save(fig, out_dir, filename) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_alcohol_by_quality(df: pd.DataFrame, out_dir) -> Path:
    """Boxplot of alcohol for each quality score; returns the saved path."""
    _check_not_empty(df)
    levels = sorted(df["quality"].unique())
    groups = [df.loc[df["quality"] == q, "alcohol"] for q in levels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(groups, tick_labels=[str(q) for q in levels])
    ax.set_xlabel("Quality score")
    ax.set_ylabel("Alcohol (% vol)")
    ax.set_title("Alcohol by quality score")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir, BOXPLOT_FILENAME)


def plot_alcohol_vs_density(df: pd.DataFrame, out_dir) -> Path:
    """Scatter of alcohol vs density with a linear trend line; returns the path."""
    _check_not_empty(df)
    x = df["alcohol"].to_numpy()
    y = df["density"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, s=8, alpha=0.3, label="Wines")
    ax.plot(x_line, slope * x_line + intercept, color="crimson", label="Linear trend")
    ax.set_xlabel("Alcohol (% vol)")
    ax.set_ylabel("Density (g/cm³)")
    ax.set_title("Alcohol vs density")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, SCATTER_FILENAME)
