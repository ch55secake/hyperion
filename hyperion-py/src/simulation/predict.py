import os

from src.simulation import predict_today


def predict_mode(visualisation: bool = False):
    """Run prediction mode for today's data"""
    print("\n" + "=" * 80)
    print("PREDICTION MODE - Make predictions on today's data")
    print("=" * 80)

    # Get symbols from saved models
    model_files = [f for f in os.listdir("models") if f.endswith("_model.pkl")]

    if not model_files:
        print("\n⚠️  No trained models found in 'models/' directory")
        print("   Please run training mode first to train models")
        return

    # Handle both stacked and normal model naming
    symbols = [f.replace("_stacked_model.pkl", "").replace("_model.pkl", "") for f in model_files]

    print(f"\nFound trained models for: {', '.join(symbols)}")
    print("\nMaking predictions for all symbols...\n")

    predictions = []
    for symbol in symbols:
        result = predict_today(symbol, visualisation=visualisation)
        if result and isinstance(result, dict):
            predictions.append(result)

    if not predictions:
        print("\n⚠️  No predictions were made. Check models or data sources.\n")
        return

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL PREDICTIONS")
    print("=" * 80)
    print(f"\n{'Symbol':<10} {'Current':<12} {'Predicted':<12} {'Return':<10} {'Signal':<20}")
    print("-" * 80)
    for p in predictions:
        print(
            f"{p['symbol']:<10} ${p['current_price']:<11.2f} ${p['predicted_price']:<11.2f} "
            f"{p['predicted_return'] * 100:>+8.3f}% {p['signal']:<20}"
        )
    print("=" * 80 + "\n")
