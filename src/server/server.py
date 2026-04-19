import re
from typing import Dict, Any

from flask import Flask, request, jsonify


class ModelServer:
    """
    Spins up a flask server that accepts /train and /predict and /tradingresults endpoints
    """

    def __init__(self, port=5000):
        self.app = Flask(__name__)
        self.port = port
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route("/trading-results/<ticker>", methods=["GET"])
        def results(ticker: str):
            try:
                if not isinstance(ticker, str) or not ticker:
                    return jsonify({"error": "ticker must be a non-empty string"}), 400

                with open(f"results/{ticker}_results.txt", "r") as f:
                    trading_results: str = f.read()

                return jsonify(
                    {
                        "status": "success",
                        "prediction_results": parse_trading_results(trading_results),
                    }
                )

            except KeyError as e:
                return jsonify({"error": f"Missing field: {str(e)}"}), 400

            except ValueError as e:
                return jsonify({"error": f"Invalid value: {str(e)}"}), 400

            except Exception as e:
                self.app.logger.error(f"Training error: {str(e)}", exc_info=True)
                return jsonify({"error": f"Internal server error: {str(e)}"}), 500

        @self.app.route("/predict/<ticker>", methods=["GET"])
        def predict(ticker: str):
            try:
                if not isinstance(ticker, str) or not ticker:
                    return jsonify({"error": "ticker must be a non-empty string"}), 400

                from src.simulation import predict_today

                result = predict_today(ticker)

                if result is None:
                    return jsonify({"error": "Prediction failed - no result returned"}), 500

                if not result:
                    return jsonify({"error": "Prediction returned empty result"}), 500

                with open(f"results/{ticker}_latest_prediction.txt", "r") as f:
                    predictions: str = f.read()

                return jsonify(
                    {
                        "status": "success",
                        "prediction_results": parse_prediction_file(predictions),
                    }
                )

            except KeyError as e:
                return jsonify({"error": f"Missing field: {str(e)}"}), 400

            except ValueError as e:
                return jsonify({"error": f"Invalid value: {str(e)}"}), 400

            except Exception as e:
                self.app.logger.error(f"Training error: {str(e)}", exc_info=True)
                return jsonify({"error": f"Internal server error: {str(e)}"}), 500

        @self.app.route("/train", methods=["POST"])
        def train():
            try:
                if not request.is_json:
                    return jsonify({"error": "Content-Type must be application/json"}), 400

                data = request.get_json()

                required_fields = ["ticker", "interval", "period"]
                missing_fields = [field for field in required_fields if field not in data]

                if missing_fields:
                    return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

                ticker = data["ticker"]
                interval = data["interval"]
                period = data["period"]

                if not isinstance(ticker, str) or not ticker:
                    return jsonify({"error": "ticker must be a non-empty string"}), 400

                if not isinstance(interval, str) or interval not in ["1m", "5m", "15m", "1h", "1d"]:
                    return jsonify({"error": "interval must be one of: 1m, 5m, 15m, 1h, 1d"}), 400

                if not isinstance(period, str):
                    return jsonify({"error": "period must be a string"}), 400

                from src.train import train_model

                result = train_model(symbols=ticker, period=period, interval=interval, visualization=False)

                if result is None:
                    return jsonify({"error": "Model training failed - no result returned"}), 500

                if not result:
                    return jsonify({"error": "Model training returned empty result"}), 500

                return jsonify({"status": "success", "result": result}), 200

            except KeyError as e:
                return jsonify({"error": f"Missing field: {str(e)}"}), 400

            except ValueError as e:
                return jsonify({"error": f"Invalid value: {str(e)}"}), 400

            except Exception as e:
                self.app.logger.error(f"Training error: {str(e)}", exc_info=True)
                return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    def run(self):
        self.app.run(host="0.0.0.0", port=self.port, debug=True)


def parse_prediction_file(content: str) -> dict:
    """Parse prediction text file into structured data"""

    result = {
        "ticker": None,
        "generated_at": None,
        "data_date": None,
        "current_analysis": {},
        "forecast_180d": {},
        "milestones": [],
    }

    ticker_match = re.search(r"Latest Prediction for (\w+)", content)
    if ticker_match:
        result["ticker"] = ticker_match.group(1)

    generated_match = re.search(r"Generated: (.+)", content)
    if generated_match:
        result["generated_at"] = generated_match.group(1).strip()

    data_date_match = re.search(r"Data Date: (.+)", content)
    if data_date_match:
        result["data_date"] = data_date_match.group(1).strip()

    current_price_match = re.search(r"Current Price:\s+\$([0-9.]+)", content)
    predicted_return_match = re.search(r"Predicted Return:\s+([+-][0-9.]+)%", content)
    predicted_price_match = re.search(r"Predicted Price:\s+\$([0-9.]+)", content)
    signal_match = re.search(r"Signal:\s+(\w+)", content)
    confidence_match = re.search(r"Confidence:\s+([0-9.]+)%", content)
    recommendation_match = re.search(r"Recommendation:\s+(.+)", content)

    if current_price_match:
        result["current_analysis"]["current_price"] = float(current_price_match.group(1))
    if predicted_return_match:
        result["current_analysis"]["predicted_return"] = float(predicted_return_match.group(1))
    if predicted_price_match:
        result["current_analysis"]["predicted_price"] = float(predicted_price_match.group(1))
    if signal_match:
        result["current_analysis"]["signal"] = signal_match.group(1)
    if confidence_match:
        result["current_analysis"]["confidence"] = float(confidence_match.group(1))
    if recommendation_match:
        result["current_analysis"]["recommendation"] = recommendation_match.group(1).strip()

    starting_price_match = re.search(r"Starting Price:\s+\$([0-9.]+)", content)
    forecast_match = re.search(r"180-Day Forecast:\s+\$([0-9.]+)", content)
    expected_change_match = re.search(r"Expected Change:\s+\$([+-]?[0-9.]+)", content)
    expected_return_match = re.search(r"Expected Return:\s+([+-]?[0-9.]+)%", content)
    confidence_range_match = re.search(r"Confidence Range:\s+\$([0-9.]+) - \$([0-9.]+)", content)

    if starting_price_match:
        result["forecast_180d"]["starting_price"] = float(starting_price_match.group(1))
    if forecast_match:
        result["forecast_180d"]["forecast_price"] = float(forecast_match.group(1))
    if expected_change_match:
        result["forecast_180d"]["expected_change"] = float(expected_change_match.group(1))
    if expected_return_match:
        result["forecast_180d"]["expected_return"] = float(expected_return_match.group(1))
    if confidence_range_match:
        result["forecast_180d"]["confidence_range"] = {
            "low": float(confidence_range_match.group(1)),
            "high": float(confidence_range_match.group(2)),
        }

    milestone_matches = re.finditer(r"Day\s+(\d+):\s+\$([0-9.]+)\s+\(([+-][0-9.]+)%\)", content)
    for match in milestone_matches:
        result["milestones"].append(
            {"day": int(match.group(1)), "price": float(match.group(2)), "return": float(match.group(3))}
        )

    return result


def parse_trading_results(content: str) -> Dict[str, Any]:
    """
    Parse XGBoost prediction results text into structured JSON

    Args:
        content: Raw text content from results file

    Returns:
        Dictionary with structured prediction data
    """

    result = {
        "ticker": None,
        "data_info": {"period": None, "total_samples": None, "train_samples": None, "test_samples": None},
        "model_performance": {"test_rmse": None, "test_mae": None, "test_r2": None},
        "trading_simulation": {
            "initial_capital": None,
            "strategies": {"directional": {}, "adaptive_threshold": {}, "hold_days": {}},
            "buy_and_hold_return": None,
            "best_strategy": {"name": None, "final_value": None, "total_return": None, "number_of_trades": None},
        },
    }

    # Extract ticker
    ticker_match = re.search(r"XGBoost Stock Prediction Results for (\w+)", content)
    if ticker_match:
        result["ticker"] = ticker_match.group(1)

    # Extract data info
    period_match = re.search(r"Data Period:\s*(\w+)", content)
    if period_match:
        result["data_info"]["period"] = period_match.group(1)

    total_samples_match = re.search(r"Total Samples:\s*(\d+)", content)
    if total_samples_match:
        result["data_info"]["total_samples"] = int(total_samples_match.group(1))

    train_samples_match = re.search(r"Train Samples:\s*(\d+)", content)
    if train_samples_match:
        result["data_info"]["train_samples"] = int(train_samples_match.group(1))

    test_samples_match = re.search(r"Test Samples:\s*(\d+)", content)
    if test_samples_match:
        result["data_info"]["test_samples"] = int(test_samples_match.group(1))

    # Extract model performance
    rmse_match = re.search(r"Test RMSE:\s*([-+]?[0-9]*\.?[0-9]+)", content)
    if rmse_match:
        result["model_performance"]["test_rmse"] = float(rmse_match.group(1))

    mae_match = re.search(r"Test MAE:\s*([-+]?[0-9]*\.?[0-9]+)", content)
    if mae_match:
        result["model_performance"]["test_mae"] = float(mae_match.group(1))

    r2_match = re.search(r"Test R²:\s*([-+]?[0-9]*\.?[0-9]+)", content)
    if r2_match:
        result["model_performance"]["test_r2"] = float(r2_match.group(1))

    # Extract initial capital
    capital_match = re.search(r"Initial Capital:\s*\$([0-9,]+\.[0-9]{2})", content)
    if capital_match:
        result["trading_simulation"]["initial_capital"] = float(capital_match.group(1).replace(",", ""))

    # Extract Directional Strategy
    dir_value_match = re.search(r"Directional Strategy:.*?Final Value:\s*\$([0-9,]+\.[0-9]{2})", content, re.DOTALL)
    dir_return_match = re.search(r"Directional Strategy:.*?Total Return:\s*([-+]?[0-9]*\.?[0-9]+)%", content, re.DOTALL)
    dir_trades_match = re.search(r"Directional Strategy:.*?Number of Trades:\s*(\d+)", content, re.DOTALL)
    dir_alpha_match = re.search(
        r"Directional Strategy:.*?Alpha vs Buy&Hold:\s*([-+]?[0-9]*\.?[0-9]+)%", content, re.DOTALL
    )

    if dir_value_match:
        result["trading_simulation"]["strategies"]["directional"]["final_value"] = float(
            dir_value_match.group(1).replace(",", "")
        )
    if dir_return_match:
        result["trading_simulation"]["strategies"]["directional"]["total_return"] = float(dir_return_match.group(1))
    if dir_trades_match:
        result["trading_simulation"]["strategies"]["directional"]["number_of_trades"] = int(dir_trades_match.group(1))
    if dir_alpha_match:
        result["trading_simulation"]["strategies"]["directional"]["alpha_vs_buy_hold"] = float(dir_alpha_match.group(1))

    # Extract Adaptive Threshold Strategy
    adapt_value_match = re.search(
        r"Adaptive Threshold Strategy:.*?Final Value:\s*\$([0-9,]+\.[0-9]{2})", content, re.DOTALL
    )
    adapt_return_match = re.search(
        r"Adaptive Threshold Strategy:.*?Total Return:\s*([-+]?[0-9]*\.?[0-9]+)%", content, re.DOTALL
    )
    adapt_trades_match = re.search(r"Adaptive Threshold Strategy:.*?Number of Trades:\s*(\d+)", content, re.DOTALL)
    adapt_alpha_match = re.search(
        r"Adaptive Threshold Strategy:.*?Alpha vs Buy&Hold:\s*([-+]?[0-9]*\.?[0-9]+)%", content, re.DOTALL
    )

    if adapt_value_match:
        result["trading_simulation"]["strategies"]["adaptive_threshold"]["final_value"] = float(
            adapt_value_match.group(1).replace(",", "")
        )
    if adapt_return_match:
        result["trading_simulation"]["strategies"]["adaptive_threshold"]["total_return"] = float(
            adapt_return_match.group(1)
        )
    if adapt_trades_match:
        result["trading_simulation"]["strategies"]["adaptive_threshold"]["number_of_trades"] = int(
            adapt_trades_match.group(1)
        )
    if adapt_alpha_match:
        result["trading_simulation"]["strategies"]["adaptive_threshold"]["alpha_vs_buy_hold"] = float(
            adapt_alpha_match.group(1)
        )

    # Extract Hold Days Strategy
    hold_value_match = re.search(r"Hold Days Strategy:.*?Final Value:\s*\$([0-9,]+\.[0-9]{2})", content, re.DOTALL)
    hold_return_match = re.search(r"Hold Days Strategy:.*?Total Return:\s*([-+]?[0-9]*\.?[0-9]+)%", content, re.DOTALL)
    hold_trades_match = re.search(r"Hold Days Strategy:.*?Number of Trades:\s*(\d+)", content, re.DOTALL)
    hold_alpha_match = re.search(
        r"Hold Days Strategy:.*?Alpha vs Buy&Hold:\s*([-+]?[0-9]*\.?[0-9]+)%", content, re.DOTALL
    )

    if hold_value_match:
        result["trading_simulation"]["strategies"]["hold_days"]["final_value"] = float(
            hold_value_match.group(1).replace(",", "")
        )
    if hold_return_match:
        result["trading_simulation"]["strategies"]["hold_days"]["total_return"] = float(hold_return_match.group(1))
    if hold_trades_match:
        result["trading_simulation"]["strategies"]["hold_days"]["number_of_trades"] = int(hold_trades_match.group(1))
    if hold_alpha_match:
        result["trading_simulation"]["strategies"]["hold_days"]["alpha_vs_buy_hold"] = float(hold_alpha_match.group(1))

    # Extract Buy & Hold Return
    buy_hold_match = re.search(r"Buy & Hold Return:\s*([-+]?[0-9]*\.?[0-9]+)%", content)
    if buy_hold_match:
        result["trading_simulation"]["buy_and_hold_return"] = float(buy_hold_match.group(1))

    # Extract Best Strategy
    best_name_match = re.search(r"Best Strategy:\s*(\w+)", content)
    if best_name_match:
        result["trading_simulation"]["best_strategy"]["name"] = best_name_match.group(1).lower()

    best_value_match = re.search(r"Best Strategy:.*?Final Value:\s*\$([0-9,]+\.[0-9]{2})", content, re.DOTALL)
    if best_value_match:
        result["trading_simulation"]["best_strategy"]["final_value"] = float(best_value_match.group(1).replace(",", ""))

    best_return_match = re.search(r"Best Strategy:.*?Total Return:\s*([-+]?[0-9]*\.?[0-9]+)%", content, re.DOTALL)
    if best_return_match:
        result["trading_simulation"]["best_strategy"]["total_return"] = float(best_return_match.group(1))

    best_trades_match = re.search(r"Best Strategy:.*?Number of Trades:\s*(\d+)", content, re.DOTALL)
    if best_trades_match:
        result["trading_simulation"]["best_strategy"]["number_of_trades"] = int(best_trades_match.group(1))

    return result
