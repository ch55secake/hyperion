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


def test_parse_args_rejects_non_positive_target_horizons(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["main.py", "--target-horizons", "5,0,10"])
    with pytest.raises(SystemExit):
        _parse_args()


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["main.py", "--test-size", "0"], id="test_size_zero"),
        pytest.param(["main.py", "--test-size", "1"], id="test_size_one"),
        pytest.param(["main.py", "--test-size", "1.5"], id="test_size_above_one"),
        pytest.param(["main.py", "--target-days", "0"], id="target_days_zero"),
        pytest.param(["main.py", "--n-trials", "0"], id="n_trials_zero"),
        pytest.param(["main.py", "--transaction-cost", "-0.1"], id="transaction_cost_negative"),
        pytest.param(
            ["main.py", "--target-up-threshold", "0.01", "--target-down-threshold", "0.02"],
            id="thresholds_inverted",
        ),
        pytest.param(["main.py", "--target-horizons", "5,-1,10"], id="negative_target_horizon"),
    ],
)
def test_parse_args_rejects_other_invalid_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        _parse_args()
