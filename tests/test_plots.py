import matplotlib.pyplot as plt
import pytest

from wine_analysis.plots import plot_alcohol_by_quality, plot_alcohol_vs_density

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

PLOT_FUNCS = [plot_alcohol_by_quality, plot_alcohol_vs_density]


@pytest.mark.parametrize("plot_func", PLOT_FUNCS)
def test_creates_non_empty_png(wine_df, tmp_path, plot_func):
    path = plot_func(wine_df, tmp_path)
    assert path.parent == tmp_path
    assert path.suffix == ".png"
    assert path.stat().st_size > 0
    assert path.read_bytes()[:8] == PNG_MAGIC


@pytest.mark.parametrize("plot_func", PLOT_FUNCS)
def test_creates_missing_nested_dir(wine_df, tmp_path, plot_func):
    out_dir = tmp_path / "a" / "b"
    assert not out_dir.exists()
    path = plot_func(wine_df, out_dir)
    assert path.exists()


@pytest.mark.parametrize("plot_func", PLOT_FUNCS)
def test_closes_figure(wine_df, tmp_path, plot_func):
    plot_func(wine_df, tmp_path)
    assert plt.get_fignums() == []


def test_filenames(wine_df, tmp_path):
    assert plot_alcohol_by_quality(wine_df, tmp_path).name == (
        "alcohol_by_quality_boxplot.png"
    )
    assert plot_alcohol_vs_density(wine_df, tmp_path).name == (
        "alcohol_vs_density_scatter.png"
    )
