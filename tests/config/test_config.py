"""Unit tests for src/config.py (HyperionConfig dataclass)."""

from src.config import HyperionConfig


class TestHyperionConfigDefaults:
    def test_default_period(self):
        cfg = HyperionConfig()
        assert cfg.period == "2y"

    def test_default_intervals(self):
        cfg = HyperionConfig()
        assert cfg.intervals == ["1d", "1h"]

    def test_default_test_size(self):
        cfg = HyperionConfig()
        assert cfg.test_size == 0.2

    def test_default_val_size(self):
        cfg = HyperionConfig()
        assert cfg.val_size == 0.1

    def test_default_target_days(self):
        cfg = HyperionConfig()
        assert cfg.target_days == 10

    def test_default_n_trials(self):
        cfg = HyperionConfig()
        assert cfg.n_trials == 1000

    def test_default_r2_save_threshold(self):
        cfg = HyperionConfig()
        assert cfg.r2_save_threshold == 0.0012

    def test_default_r2_invalid_threshold(self):
        cfg = HyperionConfig()
        assert cfg.r2_invalid_threshold == -0.3

    def test_default_initial_capital(self):
        cfg = HyperionConfig()
        assert cfg.initial_capital == 10000

    def test_default_transaction_cost(self):
        cfg = HyperionConfig()
        assert cfg.transaction_cost == 0.001


class TestHyperionConfigCustomValues:
    def test_custom_period(self):
        cfg = HyperionConfig(period="5y")
        assert cfg.period == "5y"

    def test_custom_intervals(self):
        cfg = HyperionConfig(intervals=["1d"])
        assert cfg.intervals == ["1d"]

    def test_custom_test_size(self):
        cfg = HyperionConfig(test_size=0.3)
        assert cfg.test_size == 0.3

    def test_custom_val_size(self):
        cfg = HyperionConfig(val_size=0.15)
        assert cfg.val_size == 0.15

    def test_custom_target_days(self):
        cfg = HyperionConfig(target_days=5)
        assert cfg.target_days == 5

    def test_custom_n_trials(self):
        cfg = HyperionConfig(n_trials=200)
        assert cfg.n_trials == 200

    def test_custom_r2_save_threshold(self):
        cfg = HyperionConfig(r2_save_threshold=0.005)
        assert cfg.r2_save_threshold == 0.005

    def test_custom_r2_invalid_threshold(self):
        cfg = HyperionConfig(r2_invalid_threshold=-0.5)
        assert cfg.r2_invalid_threshold == -0.5

    def test_custom_initial_capital(self):
        cfg = HyperionConfig(initial_capital=50000)
        assert cfg.initial_capital == 50000

    def test_custom_transaction_cost(self):
        cfg = HyperionConfig(transaction_cost=0.005)
        assert cfg.transaction_cost == 0.005

    def test_intervals_are_independent_between_instances(self):
        cfg1 = HyperionConfig()
        cfg2 = HyperionConfig()
        cfg1.intervals.append("4h")
        assert "4h" not in cfg2.intervals
