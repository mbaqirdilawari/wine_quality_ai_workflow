import pytest

from wine_analysis.model import FEATURES_ALL, FEATURES_BASIC, train_and_evaluate


def test_feature_sets():
    assert FEATURES_BASIC == ["alcohol", "volatile acidity", "sulphates"]
    assert len(FEATURES_ALL) == 11
    assert "type" not in FEATURES_ALL
    assert "quality" not in FEATURES_ALL


def test_exact_linear_relationship_gives_r2_one(linear_df):
    result = train_and_evaluate(linear_df, FEATURES_BASIC)
    assert result["r2"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(0.0, abs=1e-9)


def test_result_keys_and_split_sizes(linear_df):
    result = train_and_evaluate(linear_df, FEATURES_ALL)
    assert set(result) == {"features", "r2", "rmse", "n_train", "n_test"}
    assert result["features"] == FEATURES_ALL
    assert result["n_train"] + result["n_test"] == len(linear_df)
    assert result["n_test"] == 12  # 20% of 60
    assert isinstance(result["r2"], float)
    assert isinstance(result["rmse"], float)


def test_unknown_feature_raises(linear_df):
    with pytest.raises(ValueError, match="not a column"):
        train_and_evaluate(linear_df, ["alcohol", "not a column"])


def test_empty_feature_list_raises(linear_df):
    with pytest.raises(ValueError):
        train_and_evaluate(linear_df, [])


def test_too_few_rows_raises(linear_df):
    with pytest.raises(ValueError, match="rows"):
        train_and_evaluate(linear_df.head(3), FEATURES_BASIC)
