from typing import Any

from numpy import ndarray, dtype
from pandas import Series


def save_trained_model(
    predictor: dict[str, ndarray[Any, dtype[Any]] | list[Any] | dict[str, float | Any] | float | Any] | Any,
    symbol,
    test_results: Series | Any,
):
    # Save the trained model
    print("\n" + "=" * 60)
    print("Saving Model")
    print("=" * 60)
    predictor.save_model(symbol)

    # Print final test results
    print("\nFinal Test Set Performance:")
    print(f"  RMSE: {test_results['rmse']:.8f}")
    print(f"  MAE:  {test_results['mae']:.8f}")
    print(f"  R²:   {test_results['r2']:.8f}")
