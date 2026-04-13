from typing import Any

from src.util import logger


def save_trained_model(predictor: Any, symbol: str, test_results: dict):
    logger.info("=" * 60)
    logger.info("Saving Model")
    logger.info("=" * 60)

    r2 = test_results.get("r2")
    if not isinstance(r2, (float, int)):
        logger.warning(f"R\u00b2 is not a number ({type(r2)}). Skipping save.")
        return

    if r2 > 0.0012:
        predictor.save_model(symbol)

    if r2 < -0.3:
        predictor.save_model(symbol, save_path="invalid_models")

    logger.info("Final Test Set Performance:")
    logger.info(f"  RMSE: {test_results.get('rmse', float('nan')):.8f}")
    logger.info(f"  MAE:  {test_results.get('mae', float('nan')):.8f}")
    logger.info(f"  R\u00b2:   {r2:.8f}")
