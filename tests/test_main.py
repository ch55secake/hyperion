import pytest

from src.main import _parse_args


def test_parse_args_drops_empty_intervals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["main.py", "--intervals", "1d, ,1h,"])
    config = _parse_args()
    assert config.intervals == ["1d", "1h"]


def test_parse_args_rejects_empty_intervals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["main.py", "--intervals", " , , "])
    with pytest.raises(SystemExit):
        _parse_args()


def test_parse_args_rejects_invalid_test_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["main.py", "--test-size", "1"])
    with pytest.raises(SystemExit):
        _parse_args()


def test_parse_args_rejects_non_positive_target_horizons(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["main.py", "--target-horizons", "5,0,10"])
    with pytest.raises(SystemExit):
        _parse_args()
