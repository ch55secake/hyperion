"""Unit tests for src/writer/model_writer.py and src/writer/result_writer.py."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.writer.model_writer import save_trained_model
from src.writer.result_writer import output_best_strategy, persist_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_strategies(n=3):
    """Return a list of (name, results_dict, simulator_stub) tuples."""
    strategies = []
    for i in range(n):
        results = {
            "final_value": 10_000 + i * 500,
            "total_return": 0.05 * (i + 1),
            "num_trades": 5 + i,
            "buy_hold_return": 0.03,
        }
        strategies.append((f"strategy_{i}", results, MagicMock()))
    return strategies


def _make_test_results():
    return {"r2": 0.05, "rmse": 0.002, "mae": 0.001, "predictions": []}


# ---------------------------------------------------------------------------
# save_trained_model
# ---------------------------------------------------------------------------


class TestSaveTrainedModel:
    def test_saves_model_when_r2_above_threshold(self):
        predictor = MagicMock()
        save_trained_model(predictor, "TEST", {"r2": 0.002, "rmse": 0.001, "mae": 0.0005})
        predictor.save_model.assert_called_once_with("TEST")

    def test_does_not_save_when_r2_in_neutral_range(self):
        predictor = MagicMock()
        # r2=0.001 is below 0.0012 threshold and above -0.3 threshold → no save
        save_trained_model(predictor, "TEST", {"r2": 0.001, "rmse": 0.001, "mae": 0.0005})
        predictor.save_model.assert_not_called()

    def test_saves_to_invalid_models_when_r2_very_negative(self):
        predictor = MagicMock()
        save_trained_model(predictor, "TEST", {"r2": -0.5, "rmse": 0.1, "mae": 0.05})
        # Should save to invalid_models path
        predictor.save_model.assert_called_once_with("TEST", save_path="invalid_models")

    def test_skips_save_when_r2_not_numeric(self):
        predictor = MagicMock()
        save_trained_model(predictor, "TEST", {"r2": None, "rmse": 0.001, "mae": 0.0005})
        predictor.save_model.assert_not_called()

    def test_skips_save_when_r2_is_string(self):
        predictor = MagicMock()
        save_trained_model(predictor, "TEST", {"r2": "N/A", "rmse": 0.001, "mae": 0.0005})
        predictor.save_model.assert_not_called()


# ---------------------------------------------------------------------------
# output_best_strategy
# ---------------------------------------------------------------------------


class TestOutputBestStrategy:
    def test_returns_highest_return_strategy(self):
        strategies = _make_valid_strategies(3)
        best, sim_results = output_best_strategy(strategies)
        # strategy_2 has the highest total_return (0.15)
        assert best[0] == "strategy_2"

    def test_returns_tuple_of_two(self):
        strategies = _make_valid_strategies(2)
        result = output_best_strategy(strategies)
        assert len(result) == 2

    def test_sim_results_is_the_best_results_dict(self):
        strategies = _make_valid_strategies(3)
        best, sim_results = output_best_strategy(strategies)
        assert sim_results == best[1]

    def test_single_strategy_returns_that_strategy(self):
        strategies = _make_valid_strategies(1)
        best, _ = output_best_strategy(strategies)
        assert best[0] == "strategy_0"

    def test_prefers_more_trades_on_equal_return(self):
        results_a = {"total_return": 0.05, "num_trades": 2, "buy_hold_return": 0.03, "final_value": 10_500}
        results_b = {"total_return": 0.05, "num_trades": 10, "buy_hold_return": 0.03, "final_value": 10_500}
        strategies = [("a", results_a, MagicMock()), ("b", results_b, MagicMock())]
        best, _ = output_best_strategy(strategies)
        assert best[0] == "b"


# ---------------------------------------------------------------------------
# persist_results
# ---------------------------------------------------------------------------


class TestPersistResults:
    def _call_persist(self, tmpdir, strategies=None, valid_strategies=None):
        """Convenience wrapper around persist_results that writes to tmpdir."""
        symbol = "TEST"
        # Patch the results directory to use tmpdir
        results_file = os.path.join(tmpdir, f"{symbol}_results.txt")

        if strategies is None:
            strategies = _make_valid_strategies(2)
        if valid_strategies is None:
            valid_strategies = strategies

        best = max(valid_strategies, key=lambda x: x[1]["total_return"])

        x = pd.DataFrame({"Close": range(100)})
        x_train = {"daily": pd.DataFrame({"Close": range(70)})}
        x_test = {"daily": pd.DataFrame({"Close": range(30)})}
        test_results = _make_test_results()

        with patch("src.writer.result_writer.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__ = lambda s: mock_file
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            persist_results(
                x=x,
                x_test=x_test,
                x_train=x_train,
                best_strategy=best,
                period="2022-01-01 to 2023-01-01",
                sim_results=strategies[0][1],
                strategies=strategies,
                symbol=symbol,
                test_results=test_results,
                valid_strategies=valid_strategies,
            )
        return mock_file

    def test_persist_results_writes_symbol_header(self, tmp_path):  # noqa: ARG002
        mock_file = self._call_persist(str(tmp_path))
        written = "".join(str(call.args[0]) for call in mock_file.write.call_args_list)
        assert "TEST" in written

    def test_persist_results_writes_model_performance(self, tmp_path):  # noqa: ARG002
        mock_file = self._call_persist(str(tmp_path))
        written = "".join(str(call.args[0]) for call in mock_file.write.call_args_list)
        assert "R²" in written or "r2" in written.lower() or "R" in written

    def test_persist_results_writes_strategy_names(self, tmp_path):  # noqa: ARG002
        mock_file = self._call_persist(str(tmp_path))
        written = "".join(str(call.args[0]) for call in mock_file.write.call_args_list)
        assert "strategy_0" in written

    def test_persist_results_calls_open(self, tmp_path):
        """Verify that persist_results attempts to open a results file."""
        with patch("src.writer.result_writer.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__ = lambda s: mock_file
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            strategies = _make_valid_strategies(1)
            persist_results(
                x=pd.DataFrame({"Close": range(10)}),
                x_test={"daily": pd.DataFrame({"Close": range(3)})},
                x_train={"daily": pd.DataFrame({"Close": range(7)})},
                best_strategy=strategies[0],
                period="test",
                sim_results=strategies[0][1],
                strategies=strategies,
                symbol="SYM",
                test_results=_make_test_results(),
                valid_strategies=strategies,
            )
        mock_open.assert_called_once()
