from typing import Any


def save_trained_model(predictor: Any, symbol: str, test_results: dict):
    print("\n" + "=" * 60)
    print("Saving Model")
    print("=" * 60)

    r2 = test_results.get("r2")
    if not isinstance(r2, (float, int)):
        print(f"⚠️  R² is not a number ({type(r2)}). Skipping save.")
        return

    # if r2 > 0.0012:
    predictor.save_model(symbol)

    if r2 < -0.3:
        predictor.save_model(symbol, save_path="invalid_models")

    print("\nFinal Test Set Performance:")
    print(f"  RMSE: {test_results.get('rmse', float('nan')):.8f}")
    print(f"  MAE:  {test_results.get('mae', float('nan')):.8f}")
    print(f"  R²:   {r2:.8f}")
