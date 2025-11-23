import os
import warnings

from src.pipeline.stacked_pipeline import StackedModelTrainingPipeline

warnings.filterwarnings("ignore")

# Create directories for outputs
os.makedirs("./historic_data", exist_ok=True)
os.makedirs("./plots", exist_ok=True)
os.makedirs("./results", exist_ok=True)
os.makedirs("./models", exist_ok=True)
os.makedirs("./invalid_models", exist_ok=True)
os.makedirs("./params", exist_ok=True)

if __name__ == "__main__":
    stacked_pipeline: StackedModelTrainingPipeline = StackedModelTrainingPipeline()

    (stacked_pipeline.read_tickers().download_data().prepare_features().train().evaluate_model())

    # train_single_model_for_all_stocks(visualization=True)
    # from src.simulation import predict_mode
    # predict_mode(visualisation=True)
    # ModelServer(port=8080).run()
    # # Check for command line argument
    # if len(sys.argv) > 1 and sys.argv[1] == 'predict':
    #     # Prediction mode - use saved models on today's data
    #     predict_mode()
    # else:
    #     # Training mode - train new models
    #     main()
