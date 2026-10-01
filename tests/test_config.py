from pathlib import Path

from wine_analysis import config


def test_data_path_default(monkeypatch):
    monkeypatch.delenv("WINE_DATA_PATH", raising=False)
    assert config.get_data_path() == Path("data/wine_quality_merged.csv")


def test_data_path_env_override(monkeypatch, tmp_path):
    custom = tmp_path / "other.csv"
    monkeypatch.setenv("WINE_DATA_PATH", str(custom))
    assert config.get_data_path() == custom


def test_output_dir_default(monkeypatch, tmp_path):
    monkeypatch.delenv("WINE_OUTPUT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    out = config.get_output_dir()
    assert out == Path("outputs")
    assert (tmp_path / "outputs").is_dir()


def test_output_dir_env_override_creates_dir(monkeypatch, tmp_path):
    target = tmp_path / "nested" / "charts"
    monkeypatch.setenv("WINE_OUTPUT_DIR", str(target))
    assert config.get_output_dir() == target
    assert target.is_dir()


def test_empty_env_var_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("WINE_DATA_PATH", "")
    assert config.get_data_path() == Path("data/wine_quality_merged.csv")
