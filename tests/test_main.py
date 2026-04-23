"""Tests for CLI argument parsing in src/main.py."""

from src.main import _parse_args


def test_parse_args_default_val_size(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py"])

    config = _parse_args()

    assert config.val_size == 0.1


def test_parse_args_accepts_custom_val_size(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "--val-size", "0.25"])

    config = _parse_args()

    assert config.val_size == 0.25
