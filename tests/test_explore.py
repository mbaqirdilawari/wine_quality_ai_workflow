import pytest

from wine_analysis.explore import (
    filter_by_type,
    filter_min_alcohol,
    filter_min_quality,
    summary_by_quality,
    summary_by_type,
)


def test_filter_min_alcohol(wine_df):
    result = filter_min_alcohol(wine_df, 12.0)
    assert sorted(result["alcohol"]) == [12.0, 12.5, 13.0]


def test_filter_by_type(wine_df):
    result = filter_by_type(wine_df, "red")
    assert len(result) == 3
    assert set(result["type"]) == {"red"}


def test_filter_min_quality(wine_df):
    result = filter_min_quality(wine_df, 7)
    assert sorted(result["quality"]) == [7, 7, 8]


def test_filters_do_not_mutate_input(wine_df):
    before = wine_df.copy()
    filter_min_alcohol(wine_df)
    filter_by_type(wine_df)
    filter_min_quality(wine_df)
    assert wine_df.equals(before)


@pytest.mark.parametrize(
    "func, arg",
    [
        (filter_min_alcohol, 99.0),
        (filter_by_type, "rose"),
        (filter_min_quality, 10),
    ],
)
def test_filter_no_match_returns_empty(wine_df, func, arg):
    result = func(wine_df, arg)
    assert result.empty
    assert list(result.columns) == list(wine_df.columns)


def test_summary_by_type(wine_df):
    summary = summary_by_type(wine_df)
    assert list(summary.index) == ["red", "white"]
    assert summary.loc["red", "alcohol"] == pytest.approx((9.0 + 10.0 + 12.5) / 3)
    assert summary.loc["white", "alcohol"] == pytest.approx(12.0)
    assert "volatile acidity" in summary.columns


def test_summary_by_type_custom_features(wine_df):
    summary = summary_by_type(wine_df, ["density"])
    assert list(summary.columns) == ["density"]


def test_summary_by_quality(wine_df):
    summary = summary_by_quality(wine_df)
    assert list(summary.index) == [5, 6, 7, 8]
    assert list(summary.columns) == ["count", "mean_alcohol"]
    assert summary.loc[5, "count"] == 2
    assert summary.loc[5, "mean_alcohol"] == pytest.approx(9.5)
    assert summary.loc[7, "mean_alcohol"] == pytest.approx(12.25)
