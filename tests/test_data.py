import pandas as pd
import pytest

from wine_analysis.data import EXPECTED_COLUMNS, clean_data, inspect_data, load_data

# --- load_data ---


def test_load_valid_csv(tmp_path, wine_df):
    path = tmp_path / "wine.csv"
    wine_df.to_csv(path, index=False)
    df = load_data(path)
    assert df.shape == wine_df.shape
    assert list(df.columns) == EXPECTED_COLUMNS


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_data(tmp_path / "nope.csv")


def test_load_missing_column_raises(tmp_path, wine_df):
    path = tmp_path / "wine.csv"
    wine_df.drop(columns=["volatile acidity"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="volatile acidity"):
        load_data(path)


def test_load_drops_extra_columns(tmp_path, wine_df):
    # An id column would hide duplicates and an empty notes column would make
    # every row look incomplete; both must be dropped on load.
    df = pd.concat([wine_df, wine_df.iloc[[0]]], ignore_index=True)
    df.insert(0, "id", range(len(df)))
    df["notes"] = None
    path = tmp_path / "wine.csv"
    df.to_csv(path, index=False)
    loaded = load_data(path)
    assert list(loaded.columns) == EXPECTED_COLUMNS
    assert inspect_data(loaded)["duplicates"] == 1
    assert len(clean_data(loaded)) == len(wine_df)


def test_load_non_numeric_column_raises(tmp_path, wine_df):
    df = wine_df.astype({"alcohol": object})
    df.loc[0, "alcohol"] = "abc"
    path = tmp_path / "wine.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-numeric.*alcohol"):
        load_data(path)


def test_load_empty_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_data(path)


def test_load_header_only_raises(tmp_path):
    path = tmp_path / "header.csv"
    pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no data rows"):
        load_data(path)


# --- inspect_data ---


def test_inspect_known_nan_and_duplicate(dup_df):
    info = inspect_data(dup_df)
    assert info["shape"] == (4, 13)
    assert info["missing"]["chlorides"] == 1
    assert sum(info["missing"].values()) == 1
    assert info["duplicates"] == 1


def test_inspect_zero_duplicates(wine_df):
    info = inspect_data(wine_df)
    assert info["duplicates"] == 0
    assert all(n == 0 for n in info["missing"].values())


def test_inspect_does_not_print(wine_df, capsys):
    inspect_data(wine_df)
    assert capsys.readouterr().out == ""


# --- clean_data ---


def test_clean_removes_exact_duplicates_only(dup_df):
    complete = dup_df.iloc[:3]  # leave out the NaN row; tested separately below
    cleaned = clean_data(complete)
    assert len(cleaned) == 2
    assert not cleaned.duplicated().any()
    # the near-duplicate (alcohol 10.1) is kept
    assert (cleaned["alcohol"] == 10.1).sum() == 1
    assert list(cleaned.index) == [0, 1]


def test_clean_drops_rows_with_missing_values(dup_df):
    before = dup_df.copy()
    cleaned = clean_data(dup_df)
    # 4 rows: 1 exact duplicate and 1 row with a NaN removed
    assert len(cleaned) == 2
    assert cleaned.isna().sum().sum() == 0
    assert list(cleaned.index) == [0, 1]
    pd.testing.assert_frame_equal(dup_df, before)  # input not mutated


def test_clean_keeps_outliers(wine_df):
    outlier = wine_df.iloc[[0]].copy()
    outlier["residual sugar"] = 999.0
    df = pd.concat([wine_df, outlier], ignore_index=True)
    cleaned = clean_data(df)
    assert len(cleaned) == len(df)
    assert cleaned["residual sugar"].max() == 999.0


def test_clean_does_not_mutate_input(dup_df):
    before = dup_df.copy()
    clean_data(dup_df)
    pd.testing.assert_frame_equal(dup_df, before)


def test_clean_is_idempotent(dup_df):
    once = clean_data(dup_df)
    twice = clean_data(once)
    pd.testing.assert_frame_equal(once, twice)


def test_clean_empty_frame():
    empty = pd.DataFrame(columns=EXPECTED_COLUMNS)
    cleaned = clean_data(empty)
    assert cleaned.empty
    assert list(cleaned.columns) == EXPECTED_COLUMNS
