"""Unit tests for src/writer/model_writer.py."""

from unittest.mock import MagicMock, call

from src.writer.model_writer import save_trained_model


def _make_predictor():
    return MagicMock()


class TestSaveTrainedModel:
    def test_saves_model_when_r2_above_default_threshold(self):
        predictor = _make_predictor()
        save_trained_model(predictor, "AAPL", {"r2": 0.005, "rmse": 0.01, "mae": 0.008})
        predictor.save_model.assert_called_once_with("AAPL")

    def test_does_not_save_model_when_r2_below_default_threshold(self):
        predictor = _make_predictor()
        # r2=0.0005 is below the default save threshold (0.0012), so no normal save should occur
        save_trained_model(predictor, "AAPL", {"r2": 0.0005, "rmse": 0.01, "mae": 0.008})
        for c in predictor.save_model.call_args_list:
            assert c != call("AAPL")

    def test_saves_invalid_model_when_r2_below_invalid_threshold(self):
        predictor = _make_predictor()
        save_trained_model(predictor, "AAPL", {"r2": -0.5, "rmse": 0.1, "mae": 0.09})
        predictor.save_model.assert_any_call("AAPL", save_path="invalid_models")

    def test_skips_when_r2_is_not_numeric(self):
        predictor = _make_predictor()
        save_trained_model(predictor, "AAPL", {"r2": "bad", "rmse": 0.01, "mae": 0.008})
        predictor.save_model.assert_not_called()

    def test_skips_when_r2_is_missing(self):
        predictor = _make_predictor()
        save_trained_model(predictor, "AAPL", {"rmse": 0.01, "mae": 0.008})
        predictor.save_model.assert_not_called()

    def test_custom_r2_save_threshold(self):
        predictor = _make_predictor()
        # r2=0.003 is above custom threshold of 0.002, so model should be saved
        save_trained_model(predictor, "AAPL", {"r2": 0.003, "rmse": 0.01, "mae": 0.008}, r2_save_threshold=0.002)
        predictor.save_model.assert_called_once_with("AAPL")

    def test_custom_r2_save_threshold_not_met(self):
        predictor = _make_predictor()
        # r2=0.001 is below custom threshold of 0.002, so model should NOT be saved normally
        save_trained_model(predictor, "AAPL", {"r2": 0.001, "rmse": 0.01, "mae": 0.008}, r2_save_threshold=0.002)
        for c in predictor.save_model.call_args_list:
            assert c != call("AAPL")

    def test_custom_r2_invalid_threshold(self):
        predictor = _make_predictor()
        # r2=-0.1 is above default invalid threshold of -0.3, but below custom -0.05
        save_trained_model(
            predictor,
            "AAPL",
            {"r2": -0.1, "rmse": 0.1, "mae": 0.09},
            r2_invalid_threshold=-0.05,
        )
        predictor.save_model.assert_any_call("AAPL", save_path="invalid_models")

    def test_both_conditions_can_trigger(self):
        """Verifies that when r2 is above both thresholds, only normal save triggers (not invalid save)."""
        predictor = _make_predictor()
        # r2=0.002 > 0.0012 but > -0.3, so only normal save should trigger
        save_trained_model(predictor, "AAPL", {"r2": 0.002, "rmse": 0.01, "mae": 0.008})
        predictor.save_model.assert_called_once_with("AAPL")
