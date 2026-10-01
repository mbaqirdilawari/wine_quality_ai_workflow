"""Regression and integration tests on the real CSV.

The fixed counts act as data-drift alarms: if the CSV changes, they fail.
"""

from pathlib import Path

import pytest

from wine_analysis import main as main_module
from wine_analysis.data import clean_data, inspect_data, load_data
from wine_analysis.model import FEATURES_ALL, FEATURES_BASIC, train_and_evaluate

REAL_CSV = Path(__file__).resolve().parent.parent / "data" / "wine_quality_merged.csv"


@pytest.fixture(scope="module")
def raw_df():
    return load_data(REAL_CSV)


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean_data(raw_df)


# --- regression ---


def test_raw_shape_missing_duplicates(raw_df):
    info = inspect_data(raw_df)
    assert info["shape"] == (6497, 13)
    assert sum(info["missing"].values()) == 0
    assert info["duplicates"] == 1177


def test_cleaned_counts(clean_df):
    assert len(clean_df) == 5320
    assert clean_df["type"].value_counts().to_dict() == {"white": 3961, "red": 1359}
    assert not clean_df.duplicated().any()


def test_all_features_at_least_as_good_as_basic(clean_df):
    basic = train_and_evaluate(clean_df, FEATURES_BASIC)
    full = train_and_evaluate(clean_df, FEATURES_ALL)
    assert full["r2"] >= basic["r2"]


@pytest.mark.parametrize("features", [FEATURES_BASIC, FEATURES_ALL])
def test_r2_plausible_and_deterministic(clean_df, features):
    first = train_and_evaluate(clean_df, features)
    second = train_and_evaluate(clean_df, features)
    assert 0.1 <= first["r2"] <= 0.5
    assert first == second


@pytest.mark.parametrize(
    "features, r2, rmse",
    [(FEATURES_BASIC, 0.275, 0.738), (FEATURES_ALL, 0.302, 0.724)],
)
def test_readme_results_pinned(clean_df, features, r2, rmse):
    # The numbers in the README results table, to 3 decimals.
    result = train_and_evaluate(clean_df, features)
    assert result["r2"] == pytest.approx(r2, abs=5e-4)
    assert result["rmse"] == pytest.approx(rmse, abs=5e-4)
    assert (result["n_train"], result["n_test"]) == (4256, 1064)


# --- integration ---


def test_main_end_to_end(monkeypatch, tmp_path, capsys):
    out_dir = tmp_path / "charts"
    monkeypatch.setenv("WINE_DATA_PATH", str(REAL_CSV))
    monkeypatch.setenv("WINE_OUTPUT_DIR", str(out_dir))

    main_module.main()

    for name in ("alcohol_by_quality_boxplot.png", "alcohol_vs_density_scatter.png"):
        assert (out_dir / name).stat().st_size > 0

    out = capsys.readouterr().out
    for heading in ("INSPECT", "CLEANING", "EXPLORATION", "MODEL COMPARISON"):
        assert heading in out
    assert "(6497, 13)" in out
    assert "Removed 1177 exact duplicate rows" in out
    assert "Removed 0 rows with missing values" in out
    assert "-> 5320 rows remain" in out
    assert "Outliers kept" in out


def test_main_with_missing_values(monkeypatch, tmp_path, capsys, clean_df):
    # Start from the de-duplicated data so the expected counts are exact.
    df = clean_df.copy()
    df.loc[[0, 5, 9], "alcohol"] = float("nan")
    df.loc[[3], "chlorides"] = float("nan")
    csv_path = tmp_path / "with_nan.csv"
    df.to_csv(csv_path, index=False)
    out_dir = tmp_path / "charts"
    monkeypatch.setenv("WINE_DATA_PATH", str(csv_path))
    monkeypatch.setenv("WINE_OUTPUT_DIR", str(out_dir))

    main_module.main()

    for name in ("alcohol_by_quality_boxplot.png", "alcohol_vs_density_scatter.png"):
        assert (out_dir / name).stat().st_size > 0
    out = capsys.readouterr().out
    assert "Missing values (total): 4" in out
    assert "Removed 0 exact duplicate rows" in out
    assert "Removed 4 rows with missing values" in out
    assert "-> 5316 rows remain" in out
    assert "MODEL COMPARISON" in out


def test_main_no_rows_after_cleaning(monkeypatch, tmp_path, clean_df):
    df = clean_df.head(20).copy()
    df["chlorides"] = float("nan")  # every row is incomplete
    csv_path = tmp_path / "all_nan.csv"
    df.to_csv(csv_path, index=False)
    monkeypatch.setenv("WINE_DATA_PATH", str(csv_path))
    monkeypatch.setenv("WINE_OUTPUT_DIR", str(tmp_path / "out"))
    with pytest.raises(ValueError, match="No rows left after cleaning"):
        main_module.main()


def test_main_missing_data_file(monkeypatch, tmp_path):
    monkeypatch.setenv("WINE_DATA_PATH", str(tmp_path / "missing.csv"))
    monkeypatch.setenv("WINE_OUTPUT_DIR", str(tmp_path / "out"))
    with pytest.raises(FileNotFoundError):
        main_module.main()
